# Oracle Instant Client Installation for macOS

## Manual Installation Steps

### Step 1: Download Oracle Instant Client

Download the **Basic Package** from Oracle:
- **Apple Silicon (M1/M2/M3)**: https://www.oracle.com/database/technologies/instant-client/macos-arm64-downloads.html
- **Intel Mac**: https://www.oracle.com/database/technologies/instant-client/macosx-x86-64-downloads.html

**Note**: The file will be a `.dmg` file (not a ZIP), e.g., `instantclient-basic-macos.arm64-23.3.0.23.09-2.dmg`

### Step 2: Install from DMG

```bash
# Navigate to Downloads
cd ~/Downloads

# Mount the DMG file (replace with your actual filename)
hdiutil attach instantclient-basic-macos.arm64-*.dmg -quiet

# Create installation directory (no sudo needed - uses home directory)
mkdir -p ~/oracle/instantclient_23_3

# Copy files from mounted DMG to installation directory
cp -R "/Volumes/instantclient-basic-macos.arm64-"*/* ~/oracle/instantclient_23_3/

# Unmount the DMG
hdiutil detach "/Volumes/instantclient-basic-macos.arm64-"* -quiet

# Set the directory path (version may vary - check the actual directory name)
INSTANT_CLIENT_DIR="$HOME/oracle/instantclient_23_3"

# Add to your shell config
if ! grep -q "DYLD_LIBRARY_PATH.*instantclient" ~/.zshrc 2>/dev/null; then
    echo "" >> ~/.zshrc
    echo "# Oracle Instant Client" >> ~/.zshrc
    echo "export DYLD_LIBRARY_PATH=$INSTANT_CLIENT_DIR:\$DYLD_LIBRARY_PATH" >> ~/.zshrc
fi

# Reload shell config
source ~/.zshrc

# Set for current session
export DYLD_LIBRARY_PATH="$INSTANT_CLIENT_DIR:$DYLD_LIBRARY_PATH"

# Verify installation
echo "✅ Instant Client installed at: $INSTANT_CLIENT_DIR"
ls -la "$INSTANT_CLIENT_DIR" | head -5
```

### Step 3: Use in Python

```python
import oracledb
import os

# Initialize thick mode (required for Native Network Encryption)
oracledb.init_oracle_client(lib_dir=os.path.expanduser("~/oracle/instantclient_23_3"))

# Now connect to Oracle
conn = oracledb.connect(
    user=ORACLE_USER,
    password=ORACLE_PASSWORD,
    dsn=ORACLE_DSN
)
```

## Alternative: Automated Script

Use the provided `install_oracle_client.sh` script (needs to be updated for DMG support):

```bash
./install_oracle_client.sh
```

## Troubleshooting

### Architecture Mismatch Error (Most Common on macOS)

**Error Message:**
```
DatabaseError: DPI-1047: Cannot locate a 64-bit Oracle Client library:
"mach-o file, but is an incompatible architecture (have 'arm64', need 'x86_64')"
```

**Root Cause:**
This error occurs when your Python environment architecture doesn't match your Oracle Instant Client architecture. This is common on Apple Silicon Macs where:
- Your system is ARM64 (Apple Silicon)
- But your Python environment runs under x86_64 emulation (Rosetta 2)
- And you installed the ARM64 version of Oracle Instant Client

**Diagnosis Steps:**

1. **Check your Python architecture:**
   ```bash
   python -c "import platform; print(f'Python: {platform.machine()}')"
   ```
   - Output: `x86_64` = Intel/emulated
   - Output: `arm64` = Apple Silicon native

2. **Check your system architecture:**
   ```bash
   uname -m
   ```
   - Output: `arm64` = Apple Silicon
   - Output: `x86_64` = Intel Mac

3. **Check your Oracle Instant Client architecture:**
   ```bash
   file ~/oracle/instantclient_*/libclntsh.dylib.* | head -1
   ```
   - Look for `arm64` or `x86_64` in the output

**Solutions:**

**Option 1: Install Matching Architecture (Recommended)**
- If Python is `x86_64`, install x86_64 Oracle Instant Client:
  ```bash
  # Download from: https://www.oracle.com/database/technologies/instant-client/macosx-x86-64-downloads.html
  # Install to: ~/oracle/instantclient_19_16_x86_64 (or similar)
  ```
- If Python is `arm64`, install ARM64 Oracle Instant Client:
  ```bash
  # Download from: https://www.oracle.com/database/technologies/instant-client/macos-arm64-downloads.html
  # Install to: ~/oracle/instantclient_23_3 (or similar)
  ```

**Option 2: Switch Python Environment Architecture**
- Create a native ARM64 conda environment:
  ```bash
  CONDA_SUBDIR=osx-arm64 conda create -n memorizz_prod_arm64 python=3.11
  conda activate memorizz_prod_arm64
  # Install packages...
  ```
- Then use the ARM64 Oracle Instant Client

**Example Fix:**
```python
# If your Python is x86_64, use x86_64 client:
oracledb.init_oracle_client(
    lib_dir=os.path.expanduser("~/oracle/instantclient_19_16_x86_64")
)

# If your Python is arm64, use ARM64 client:
oracledb.init_oracle_client(
    lib_dir=os.path.expanduser("~/oracle/instantclient_23_3")
)
```

---

### Library Not Found Error

**Error Message:**
```
DatabaseError: DPI-1047: Cannot locate a 64-bit Oracle Client library:
"dlopen(libclntsh.dylib, 0x0001): tried: 'libclntsh.dylib' (no such file)..."
```

**Root Cause:**
- `oracledb.init_oracle_client()` was called without the `lib_dir` parameter
- Python can't auto-detect the library location
- `DYLD_LIBRARY_PATH` may not be inherited by Jupyter notebooks

**Solution:**
Always explicitly specify the `lib_dir` parameter:
```python
import oracledb
import os

# ✅ Always specify lib_dir explicitly
oracledb.init_oracle_client(
    lib_dir=os.path.expanduser("~/oracle/instantclient_23_3")
)
```

---

### Native Network Encryption Error

**Error Message:**
```
NotSupportedError: DPY-3001: Native Network Encryption and Data Integrity is only
supported in python-oracledb thick mode
```

**Root Cause:**
- Oracle server requires encryption, but Python is using thin mode (default)
- Thin mode doesn't support Native Network Encryption

**Solution:**
Initialize thick mode before connecting:
```python
import oracledb
import os

# Initialize thick mode first
oracledb.init_oracle_client(
    lib_dir=os.path.expanduser("~/oracle/instantclient_23_3")
)

# Then connect
conn = oracledb.connect(user=USER, password=PASSWORD, dsn=DSN)
```

---

### Quick Diagnostic Commands

Run these to diagnose your setup:

```bash
# 1. Check Python architecture
python -c "import platform; print(f'Python: {platform.machine()}')"

# 2. Check system architecture
uname -m

# 3. Check Oracle Instant Client architecture
file ~/oracle/instantclient_*/libclntsh.dylib.* 2>/dev/null | head -1

# 4. Verify library exists
ls -la ~/oracle/instantclient_*/libclntsh.dylib

# 5. Check environment variable (if set)
echo $DYLD_LIBRARY_PATH
```

## Notes

- **No sudo required**: Installation uses `~/oracle/` instead of `/opt/oracle/`
- **DMG format**: Oracle provides `.dmg` files for macOS, not `.zip`
- **Version numbers**: The directory name will match the version (e.g., `instantclient_23_3` for version 23.3.0.23.09-2)
- **Thick mode**: Required for Native Network Encryption and Data Integrity features
- **Architecture matching**: Python and Oracle Instant Client architectures must match (both x86_64 or both arm64)
- **Always specify lib_dir**: Don't rely on auto-detection, especially in Jupyter notebooks
