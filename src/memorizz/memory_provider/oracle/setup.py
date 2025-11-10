"""
Oracle Database Setup Module for Memorizz

This module provides the setup_oracle_user function that can be imported
and used programmatically or via CLI.

Supports two modes:
- Admin mode: Full setup with user creation and privilege grants (for local/self-hosted databases)
- User-only mode: Uses existing schema, skips user creation (for hosted databases like FreeSQL.com)
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    import oracledb
except ImportError:
    oracledb = None


def parse_sql_file(filepath):
    """Parse SQL file into individual executable statements."""
    with open(filepath, "r") as f:
        content = f.read()

    # Remove single-line comments
    lines = []
    for line in content.split("\n"):
        if "--" in line:
            line = line[: line.index("--")]
        lines.append(line)
    content = "\n".join(lines)

    # Remove multi-line comments
    while "/*" in content:
        start = content.index("/*")
        end = content.index("*/", start) + 2
        content = content[:start] + content[end:]

    # Split by semicolons
    statements = [s.strip() for s in content.split(";") if s.strip()]

    # Filter out COMMENT statements
    statements = [s for s in statements if not s.upper().startswith("COMMENT")]

    return statements


def _can_create_users(conn) -> bool:
    """
    Check if the current connection has CREATE USER privilege.

    Args:
        conn: Oracle database connection

    Returns:
        True if user can create other users, False otherwise
    """
    cursor = conn.cursor()
    try:
        # Check for CREATE USER system privilege
        cursor.execute(
            """
            SELECT COUNT(*) FROM user_sys_privs
            WHERE privilege = 'CREATE USER'
        """
        )
        result = cursor.fetchone()
        return result[0] > 0 if result else False
    except Exception:
        # If we can't query privileges, assume we can't create users
        return False
    finally:
        cursor.close()


def _check_user_privileges(conn) -> Dict[str, bool]:
    """
    Check what privileges the current user has.

    Args:
        conn: Oracle database connection

    Returns:
        Dictionary mapping privilege names to availability
    """
    cursor = conn.cursor()
    privileges = {
        "CREATE_TABLE": False,
        "CREATE_VIEW": False,
        "CREATE_INDEX": False,
        "CREATE_SEQUENCE": False,
        "CREATE_TRIGGER": False,
        "DBMS_VECTOR": False,
        "SODA_APP": False,
    }

    try:
        # Check system privileges
        cursor.execute(
            """
            SELECT privilege FROM user_sys_privs
            WHERE privilege IN ('CREATE TABLE', 'CREATE VIEW', 'CREATE INDEX',
                               'CREATE SEQUENCE', 'CREATE TRIGGER')
        """
        )
        sys_privs = {row[0] for row in cursor.fetchall()}

        privileges["CREATE_TABLE"] = "CREATE TABLE" in sys_privs
        privileges["CREATE_VIEW"] = "CREATE VIEW" in sys_privs
        privileges["CREATE_INDEX"] = "CREATE INDEX" in sys_privs
        privileges["CREATE_SEQUENCE"] = "CREATE SEQUENCE" in sys_privs
        privileges["CREATE_TRIGGER"] = "CREATE TRIGGER" in sys_privs

        # Check role privileges
        cursor.execute(
            """
            SELECT granted_role FROM user_role_privs
            WHERE granted_role = 'SODA_APP'
        """
        )
        roles = {row[0] for row in cursor.fetchall()}
        privileges["SODA_APP"] = "SODA_APP" in roles

        # Check if DBMS_VECTOR is accessible (try to describe it)
        try:
            cursor.execute(
                "SELECT COUNT(*) FROM all_objects WHERE object_name = 'DBMS_VECTOR' AND owner = 'SYS'"
            )
            result = cursor.fetchone()
            privileges["DBMS_VECTOR"] = result[0] > 0 if result else False
        except Exception:
            privileges["DBMS_VECTOR"] = False

    except Exception:
        # If we can't check, assume minimal privileges
        pass
    finally:
        cursor.close()

    return privileges


def _check_admin_capabilities(
    admin_user: str, admin_password: str, dsn: str
) -> Tuple[bool, Optional[object], Dict[str, bool]]:
    """
    Check if admin connection is possible and what capabilities are available.

    Args:
        admin_user: Admin username
        admin_password: Admin password
        dsn: Database connection string

    Returns:
        Tuple of (can_connect, connection_object, capabilities_dict)
        If connection fails, returns (False, None, {})
    """
    try:
        admin_conn = oracledb.connect(user=admin_user, password=admin_password, dsn=dsn)
        can_create = _can_create_users(admin_conn)
        capabilities = {
            "can_create_users": can_create,
            "can_grant_privileges": can_create,  # Assumed if can create users
        }
        return True, admin_conn, capabilities
    except Exception:
        return False, None, {}


def _create_user_and_grant_privileges(
    admin_conn, memorizz_user: str, memorizz_password: str, dsn: str
) -> bool:
    """
    Create user and grant all required privileges (Admin mode only).

    Args:
        admin_conn: Admin database connection
        memorizz_user: Username to create
        memorizz_password: Password for new user
        dsn: Database connection string

    Returns:
        True if successful, False otherwise
    """
    admin_cursor = admin_conn.cursor()

    try:
        # Drop existing user if exists
        print(f"\nDropping existing {memorizz_user} user (if exists)...")
        try:
            # First, kill any active sessions for this user
            print("  Checking for active sessions...")
            admin_cursor.execute(
                f"""
                SELECT sid, serial# FROM v$session WHERE username = UPPER('{memorizz_user}')
            """
            )
            sessions = admin_cursor.fetchall()

            if sessions:
                print(f"  Found {len(sessions)} active session(s), terminating...")
                for sid, serial in sessions:
                    try:
                        admin_cursor.execute(
                            f"ALTER SYSTEM KILL SESSION '{sid},{serial}' IMMEDIATE"
                        )
                        print(f"    ✓ Killed session {sid},{serial}")
                    except Exception as e:
                        error_msg = str(e)
                        if "ORA-00031" in error_msg:
                            print(f"    ✓ Session {sid},{serial} marked for kill")
                        else:
                            print(f"    ⚠ Failed to kill session {sid},{serial}: {e}")

                # Reconnect if our connection was lost
                try:
                    admin_cursor.execute("SELECT 1 FROM DUAL")
                except Exception:
                    print("  Reconnecting to database...")
                    admin_cursor.close()
                    admin_conn.close()
                    admin_conn = oracledb.connect(
                        user=os.environ.get("ORACLE_ADMIN_USER", "system"),
                        password=os.environ.get(
                            "ORACLE_ADMIN_PASSWORD", "MyPassword123!"
                        ),
                        dsn=dsn,
                    )
                    admin_cursor = admin_conn.cursor()
                    print("  ✓ Reconnected successfully")
            else:
                print("  No active sessions found")

            # Now drop the user
            admin_cursor.execute(f"DROP USER {memorizz_user} CASCADE")
            print(f"  ✓ Dropped existing {memorizz_user} user")
        except Exception as e:
            if "ORA-01918" in str(e):
                print("  ℹ User doesn't exist yet (this is fine)")
            else:
                print(f"  ⚠ {e}")

        # Create user
        print(f"\nCreating {memorizz_user} user...")
        admin_cursor.execute(
            f'CREATE USER {memorizz_user} IDENTIFIED BY "{memorizz_password}"'
        )
        print(f"  ✓ User {memorizz_user} created")

        # Grant basic privileges (least-privilege principle)
        print("\nGranting basic privileges (least-privilege)...")
        admin_cursor.execute(f"GRANT CREATE SESSION TO {memorizz_user}")
        print("  ✓ CREATE SESSION (required for database connections)")

        admin_cursor.execute(f"GRANT CREATE TABLE TO {memorizz_user}")
        print("  ✓ CREATE TABLE (required for memory storage tables)")

        admin_cursor.execute(f"GRANT CREATE INDEX TO {memorizz_user}")
        print("  ✓ CREATE INDEX (required for vector indexes)")

        admin_cursor.execute(f"GRANT CREATE VIEW TO {memorizz_user}")
        print("  ✓ CREATE VIEW (required for JSON Duality Views)")

        admin_cursor.execute(f"GRANT CREATE SEQUENCE TO {memorizz_user}")
        print("  ✓ CREATE SEQUENCE (required for ID generation)")

        admin_cursor.execute(f"GRANT CREATE TRIGGER TO {memorizz_user}")
        print("  ✓ CREATE TRIGGER (required for Duality View triggers)")

        admin_cursor.execute(f"GRANT UNLIMITED TABLESPACE TO {memorizz_user}")
        print("  ✓ UNLIMITED TABLESPACE (required for data storage)")

        # Grant AI Vector Search privileges (Oracle 23ai+)
        print("\nGranting AI Vector Search privileges...")
        try:
            admin_cursor.execute(f"GRANT EXECUTE ON DBMS_VECTOR TO {memorizz_user}")
            print("  ✓ EXECUTE ON DBMS_VECTOR (required for vector operations)")

            admin_cursor.execute(
                f"GRANT EXECUTE ON DBMS_VECTOR_CHAIN TO {memorizz_user}"
            )
            print("  ✓ EXECUTE ON DBMS_VECTOR_CHAIN (required for vector chains)")
        except Exception as e:
            print(f"  ⚠ Vector privileges not available: {e}")
            print("    This is OK if not using Oracle 23ai+")

        # Grant JSON Duality View privileges
        print("\nGranting JSON Duality View privileges...")
        try:
            admin_cursor.execute(f"GRANT SODA_APP TO {memorizz_user}")
            print("  ✓ SODA_APP (required for JSON Duality Views)")
            print(
                "  ℹ SELECT ANY TABLE removed (not required, follows least-privilege)"
            )
        except Exception as e:
            print(f"  ⚠ Failed to grant Duality View privileges: {e}")
            print("    Some features may not be available")

        admin_conn.commit()
        print("\n✓ User creation and privilege grants complete!")
        return True

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        admin_conn.rollback()
        return False
    finally:
        admin_cursor.close()


def _create_schema(user_conn, schema_file: Path) -> Tuple[int, int, int]:
    """
    Create relational schema tables.

    Args:
        user_conn: User database connection
        schema_file: Path to schema SQL file

    Returns:
        Tuple of (success_count, skip_count, fail_count)
    """
    print("\nExecuting schema SQL...")
    statements = parse_sql_file(schema_file)
    print(f"Found {len(statements)} SQL statements")

    user_cursor = user_conn.cursor()
    success_count = 0
    skip_count = 0
    fail_count = 0

    for i, stmt in enumerate(statements, 1):
        try:
            user_cursor.execute(stmt)
            success_count += 1
            if i % 10 == 0 or i == len(statements):
                print(f"  ✓ Executed {i}/{len(statements)} statements...")
        except Exception as e:
            error_str = str(e)
            if "ORA-00955" in error_str or "ORA-01408" in error_str:
                skip_count += 1
            else:
                fail_count += 1
                if fail_count <= 3:  # Only show first 3 errors
                    print(f"  ✗ Statement {i}: {e}")

    user_conn.commit()
    user_cursor.close()

    print(
        f"\n📊 Schema Summary: ✓ {success_count} success, ⚠ {skip_count} skipped, ✗ {fail_count} failed"
    )

    return success_count, skip_count, fail_count


def _create_duality_views(user_conn, views_file: Path) -> Tuple[int, int]:
    """
    Create JSON Duality Views.

    Args:
        user_conn: User database connection
        views_file: Path to views SQL file

    Returns:
        Tuple of (success_count, fail_count)
    """
    print("\nExecuting Duality Views SQL...")
    statements = parse_sql_file(views_file)
    print(f"Found {len(statements)} view statements")

    user_cursor = user_conn.cursor()
    success_count = 0
    fail_count = 0

    for i, stmt in enumerate(statements, 1):
        try:
            user_cursor.execute(stmt)
            success_count += 1
            print(f"  ✓ Created view {i}/{len(statements)}")
        except Exception as e:
            fail_count += 1
            print(f"  ✗ View {i} failed: {e}")

    user_conn.commit()
    user_cursor.close()

    print(f"\n📊 Views Summary: ✓ {success_count} success, ✗ {fail_count} failed")

    return success_count, fail_count


def _verify_setup(user_conn) -> Tuple[list, list, list]:
    """
    Verify that setup completed successfully.

    Args:
        user_conn: User database connection

    Returns:
        Tuple of (tables_list, views_list, indexes_list)
    """
    user_cursor = user_conn.cursor()

    # Check tables
    print("\n📊 Relational Tables:")
    user_cursor.execute(
        """
        SELECT table_name FROM user_tables
        WHERE table_name IN ('AGENTS', 'AGENT_LLM_CONFIGS', 'AGENT_MEMORIES', 'PERSONAS',
                             'TOOLBOX', 'CONVERSATION_MEMORY', 'LONG_TERM_MEMORY',
                             'SHORT_TERM_MEMORY', 'WORKFLOW_MEMORY', 'SHARED_MEMORY',
                             'SUMMARIES', 'SEMANTIC_CACHE')
        ORDER BY table_name
    """
    )
    tables = [row[0] for row in user_cursor.fetchall()]
    for table in tables:
        print(f"  ✓ {table}")
    print(f"  Total: {len(tables)} tables")

    # Check views
    print("\n📄 JSON Duality Views:")
    user_cursor.execute(
        "SELECT view_name FROM user_views WHERE view_name LIKE '%_DV' ORDER BY view_name"
    )
    views = [row[0] for row in user_cursor.fetchall()]
    for view in views:
        print(f"  ✓ {view}")
    print(f"  Total: {len(views)} views")

    # Check vector indexes
    print("\n🔍 Vector Indexes:")
    user_cursor.execute(
        "SELECT index_name FROM user_indexes WHERE index_name LIKE 'IDX_%_VEC' ORDER BY index_name"
    )
    indexes = [row[0] for row in user_cursor.fetchall()]
    print(f"  Total: {len(indexes)} vector indexes")

    user_cursor.close()

    return tables, views, indexes


def setup_oracle_user():
    """
    Create and configure memorizz user in Oracle database.

    Supports two modes:
    - Admin mode: Full setup with user creation (for local/self-hosted databases)
    - User-only mode: Uses existing schema (for hosted databases like FreeSQL.com)

    The function automatically detects which mode to use based on available capabilities.
    """
    if oracledb is None:
        print("✗ oracledb package not found. Install with: pip install oracledb")
        return False

    # Configuration - can be overridden via environment variables
    ADMIN_USER = os.environ.get("ORACLE_ADMIN_USER", "system")
    ADMIN_PASSWORD = os.environ.get("ORACLE_ADMIN_PASSWORD", "MyPassword123!")
    DSN = os.environ.get("ORACLE_DSN", "localhost:1521/FREEPDB1")

    # User to create/use - can be overridden via environment variables
    MEMORIZZ_USER = os.environ.get("ORACLE_USER", "memorizz_user")
    MEMORIZZ_PASSWORD = os.environ.get("ORACLE_PASSWORD", "SecurePass123!")

    # SQL files - resolve paths relative to this package
    PACKAGE_DIR = Path(__file__).parent
    SCHEMA_FILE = PACKAGE_DIR / "schema_relational.sql"
    VIEWS_FILE = PACKAGE_DIR / "duality_views.sql"

    print("=" * 70)
    print("Oracle Database Complete Setup for Memorizz")
    print("=" * 70)
    print()

    # Verify SQL files exist
    if not SCHEMA_FILE.exists():
        print(f"✗ Schema file not found: {SCHEMA_FILE}")
        print("  Please verify the package installation is correct")
        return False

    if not VIEWS_FILE.exists():
        print(f"✗ Views file not found: {VIEWS_FILE}")
        print("  Please verify the package installation is correct")
        return False

    print(f"✓ Found schema file: {SCHEMA_FILE.name}")
    print(f"✓ Found views file: {VIEWS_FILE.name}")
    print()

    # ========== DETECT SETUP MODE ==========
    print("Detecting setup mode...")
    print("-" * 70)

    # Try to connect as admin first
    can_connect_admin, admin_conn, admin_capabilities = _check_admin_capabilities(
        ADMIN_USER, ADMIN_PASSWORD, DSN
    )

    setup_mode = None
    user_conn = None

    if can_connect_admin and admin_capabilities.get("can_create_users", False):
        # Admin mode: Full setup possible
        setup_mode = "admin"
        print("✓ Admin mode detected: Full setup with user creation")
        print(f"  Connected as: {ADMIN_USER}")
        print(f"  Can create users: Yes")
    else:
        # User-only mode: Use existing schema
        setup_mode = "user_only"
        print("ℹ User-only mode detected: Using existing schema")
        if not can_connect_admin:
            print(f"  Admin connection failed (this is OK for hosted databases)")
        else:
            print(f"  Admin user '{ADMIN_USER}' cannot create users")
        print(f"  Will use existing user: {MEMORIZZ_USER}")

        # Try to connect as the regular user
        try:
            user_conn = oracledb.connect(
                user=MEMORIZZ_USER, password=MEMORIZZ_PASSWORD, dsn=DSN
            )
            print(f"  ✓ Connected as {MEMORIZZ_USER}")

            # Check user privileges
            user_privs = _check_user_privileges(user_conn)
            print("\n  User privileges:")
            for priv, available in user_privs.items():
                status = "✓" if available else "✗"
                print(f"    {status} {priv}")

            # Warn about missing privileges
            missing = [k for k, v in user_privs.items() if not v]
            if missing:
                print(f"\n  ⚠ Missing privileges: {', '.join(missing)}")
                print("    Some features may not be available")
        except Exception as e:
            print(f"  ✗ Failed to connect as {MEMORIZZ_USER}: {e}")
            print("\nPlease check:")
            print("  1. ORACLE_USER environment variable is set correctly")
            print("  2. ORACLE_PASSWORD environment variable is set correctly")
            print("  3. ORACLE_DSN environment variable is set correctly")
            if admin_conn:
                admin_conn.close()
            return False

    print()

    # ========== STEP 1: Create User (Admin Mode Only) ==========
    if setup_mode == "admin":
        print("STEP 1: Creating User and Granting Privileges")
        print("-" * 70)

        success = _create_user_and_grant_privileges(
            admin_conn, MEMORIZZ_USER, MEMORIZZ_PASSWORD, DSN
        )

        if not success:
            admin_conn.close()
            return False

        admin_conn.close()
        print()

    # ========== STEP 2: Create Schema ==========
    print("STEP 2: Creating Relational Schema")
    print("-" * 70)

    # Connect as the user (if not already connected)
    if user_conn is None:
        print(f"Connecting as {MEMORIZZ_USER}...")
        try:
            user_conn = oracledb.connect(
                user=MEMORIZZ_USER, password=MEMORIZZ_PASSWORD, dsn=DSN
            )
            print("✓ Connected!")
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False
    else:
        print(f"Using existing connection as {MEMORIZZ_USER}...")

    print(f"\nExecuting {SCHEMA_FILE.name}...")
    success_count, skip_count, fail_count = _create_schema(user_conn, SCHEMA_FILE)
    print()

    # ========== STEP 3: Create Duality Views ==========
    print("STEP 3: Creating JSON Duality Views")
    print("-" * 70)

    print(f"Executing {VIEWS_FILE.name}...")
    view_success, view_fail = _create_duality_views(user_conn, VIEWS_FILE)
    print()

    # ========== STEP 4: Verify Setup ==========
    print("STEP 4: Verifying Setup")
    print("-" * 70)

    tables, views, indexes = _verify_setup(user_conn)
    print()

    # ========== SUMMARY ==========
    print("=" * 70)
    print("✅ Setup Complete!")
    print("=" * 70)
    print()
    print("Connection details for your Memorizz application:")
    print(f"  User:     {MEMORIZZ_USER}")
    print(f"  Password: {MEMORIZZ_PASSWORD}")
    print(f"  DSN:      {DSN}")
    print()

    if setup_mode == "user_only":
        print("ℹ Setup Mode: User-only (existing schema)")
        print("  Some admin-granted privileges may not be available.")
        print("  Contact your database administrator if you need:")
        print("    - DBMS_VECTOR execute privileges (for vector search)")
        print("    - SODA_APP role (for JSON Duality Views)")
    else:
        print("ℹ Setup Mode: Admin (full setup)")

    print()
    print("Example usage in Python:")
    print()
    print("  from memorizz.memory_provider.oracle import OracleProvider, OracleConfig")
    print("  from memorizz.memagent.builders import MemAgentBuilder")
    print("  ")
    print("  # Create Oracle provider")
    print("  oracle_config = OracleConfig(")
    print(f'      user="{MEMORIZZ_USER}",')
    print(f'      password="{MEMORIZZ_PASSWORD}",')
    print(f'      dsn="{DSN}",')
    print('      embedding_provider="openai",')
    print(
        '      embedding_config={"model": "text-embedding-3-small", "api_key": "your-key"}'
    )
    print("  )")
    print("  oracle_provider = OracleProvider(oracle_config)")
    print("  ")
    print("  # Build your agent")
    print("  agent = (MemAgentBuilder()")
    print('      .with_instruction("You are a helpful assistant.")')
    print("      .with_memory_provider(oracle_provider)")
    print("      .with_llm_config({")
    print('          "provider": "openai",')
    print('          "model": "gpt-4o-mini",')
    print('          "api_key": "your-openai-key"')
    print("      })")
    print("      .build()")
    print("  )")
    print()
    print("Summary counts:")
    print(f"  - Tables: {len(tables)}/12 expected")
    print(f"  - Duality Views: {len(views)}/10 expected")
    print(f"  - Vector Indexes: {len(indexes)}/10 expected")

    if len(tables) >= 12 and len(views) >= 10:
        print("\n✅ All components created successfully!")
    else:
        print("\n⚠️  Some components may be missing. Check the logs above.")

    user_conn.close()
    print()

    return True
