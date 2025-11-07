-- ============================================================================
-- Memorizz Oracle AI Database Setup Script
-- ============================================================================
-- This script creates the necessary database user and grants required
-- privileges for Memorizz to work with Oracle AI Database 23ai/26ai
--
-- Prerequisites:
-- - Oracle Database 23ai or later
-- - Connected as SYSTEM, SYS, or a user with DBA privileges
-- - AI Vector Search feature available
--
-- Usage:
--   sqlplus system/password@//localhost:1521/FREEPDB1 @setup.sql
--   or
--   SQL> @/path/to/setup.sql
-- ============================================================================

-- Configuration variables (modify as needed)
DEFINE memorizz_user = 'memorizz_user'
DEFINE memorizz_password = 'ChangeMe_SecurePassword123!'
DEFINE default_tablespace = 'USERS'
DEFINE temp_tablespace = 'TEMP'

PROMPT
PROMPT ============================================================================
PROMPT Creating Memorizz Database User and Granting Privileges
PROMPT ============================================================================
PROMPT

-- Drop user if exists (uncomment to recreate)
-- DROP USER &memorizz_user CASCADE;

-- Create the user
PROMPT Creating user: &memorizz_user
CREATE USER &memorizz_user IDENTIFIED BY "&memorizz_password"
    DEFAULT TABLESPACE &default_tablespace
    TEMPORARY TABLESPACE &temp_tablespace
    QUOTA UNLIMITED ON &default_tablespace;

PROMPT User created successfully.
PROMPT

-- Grant basic privileges
PROMPT Granting basic privileges...
GRANT CREATE SESSION TO &memorizz_user;
GRANT CREATE TABLE TO &memorizz_user;
GRANT CREATE INDEX TO &memorizz_user;
GRANT CREATE VIEW TO &memorizz_user;
GRANT CREATE SEQUENCE TO &memorizz_user;
GRANT CREATE PROCEDURE TO &memorizz_user;

PROMPT Basic privileges granted.
PROMPT

-- Grant vector-specific privileges (Oracle 23ai+)
PROMPT Granting AI Vector Search privileges...
BEGIN
    EXECUTE IMMEDIATE 'GRANT EXECUTE ON DBMS_VECTOR TO &memorizz_user';
    EXECUTE IMMEDIATE 'GRANT EXECUTE ON DBMS_VECTOR_CHAIN TO &memorizz_user';
    DBMS_OUTPUT.PUT_LINE('Vector privileges granted.');
EXCEPTION
    WHEN OTHERS THEN
        DBMS_OUTPUT.PUT_LINE('Warning: Could not grant vector privileges.');
        DBMS_OUTPUT.PUT_LINE('Error: ' || SQLERRM);
        DBMS_OUTPUT.PUT_LINE('This is normal if not using Oracle 23ai+ or vector features not installed.');
END;
/

PROMPT

-- Grant JSON privileges
PROMPT Granting JSON privileges...
GRANT EXECUTE ON DBMS_JSON TO &memorizz_user;

PROMPT JSON privileges granted.
PROMPT

-- Create a test to verify vector support
PROMPT
PROMPT ============================================================================
PROMPT Verifying Vector Support
PROMPT ============================================================================
PROMPT

-- Switch to the new user context for testing
CONNECT &memorizz_user/"&memorizz_password"

PROMPT Testing VECTOR datatype...
SET SERVEROUTPUT ON

DECLARE
    v_test_vector VECTOR;
    v_test_passed BOOLEAN := FALSE;
BEGIN
    -- Create a simple test vector
    v_test_vector := TO_VECTOR('[1.0, 2.0, 3.0]', 3, FLOAT32);

    -- Create a temporary test table
    EXECUTE IMMEDIATE 'CREATE TABLE memorizz_vector_test (
        id NUMBER PRIMARY KEY,
        test_vector VECTOR(3, FLOAT32)
    )';

    -- Insert test data
    EXECUTE IMMEDIATE 'INSERT INTO memorizz_vector_test VALUES (1, :1)'
        USING v_test_vector;

    COMMIT;

    -- Verify we can query it
    EXECUTE IMMEDIATE 'SELECT COUNT(*) FROM memorizz_vector_test'
        INTO v_test_passed;

    -- Clean up
    EXECUTE IMMEDIATE 'DROP TABLE memorizz_vector_test PURGE';

    IF v_test_passed THEN
        DBMS_OUTPUT.PUT_LINE('✓ Vector support verified successfully!');
        DBMS_OUTPUT.PUT_LINE('  Your database supports VECTOR datatype and operations.');
    END IF;

EXCEPTION
    WHEN OTHERS THEN
        DBMS_OUTPUT.PUT_LINE('✗ Vector support test failed!');
        DBMS_OUTPUT.PUT_LINE('  Error: ' || SQLERRM);
        DBMS_OUTPUT.PUT_LINE('  Please ensure you are using Oracle Database 23ai or later.');
        DBMS_OUTPUT.PUT_LINE('  And that AI Vector Search feature is installed.');

        -- Try to clean up
        BEGIN
            EXECUTE IMMEDIATE 'DROP TABLE memorizz_vector_test PURGE';
        EXCEPTION
            WHEN OTHERS THEN NULL;
        END;
END;
/

PROMPT
PROMPT ============================================================================
PROMPT Setup Complete!
PROMPT ============================================================================
PROMPT
PROMPT User: &memorizz_user
PROMPT Password: &memorizz_password
PROMPT
PROMPT Connection String Examples:
PROMPT   DSN Format: localhost:1521/FREEPDB1
PROMPT   TNS Format: (DESCRIPTION=(ADDRESS=(PROTOCOL=TCP)(HOST=localhost)(PORT=1521))(CONNECT_DATA=(SERVICE_NAME=FREEPDB1)))
PROMPT
PROMPT Next Steps:
PROMPT   1. Update your password if using default
PROMPT   2. Test connection: python -m oracledb --help
PROMPT   3. Configure Memorizz with these credentials
PROMPT   4. Run your first agent!
PROMPT
PROMPT Example Python Configuration:
PROMPT   from memorizz.memory_provider.oracle import OracleConfig, OracleProvider
PROMPT   config = OracleConfig(
PROMPT       user='&memorizz_user',
PROMPT       password='your_password',
PROMPT       dsn='localhost:1521/FREEPDB1'
PROMPT   )
PROMPT   provider = OracleProvider(config)
PROMPT
PROMPT ============================================================================

-- Reconnect as admin for any remaining tasks
-- CONNECT system/password@//localhost:1521/FREEPDB1

PROMPT
PROMPT Setup script completed.
PROMPT
