#!/bin/bash
# Simple Oracle Instant Client Installer for macOS
# Usage: ./install_oracle_client.sh

set -e

INSTALL_DIR="$HOME/oracle/instantclient"
DOWNLOADS_DIR="$HOME/Downloads"

echo "🔍 Looking for Oracle Instant Client ZIP in Downloads..."

# Find the downloaded ZIP file
ZIP_FILE=$(find "$DOWNLOADS_DIR" -name "instantclient-basic-macos.arm64-*.zip" -o -name "instantclient-basic-macosx.x86-64-*.zip" 2>/dev/null | head -1)

if [ -z "$ZIP_FILE" ]; then
    echo "❌ Oracle Instant Client ZIP not found in Downloads folder"
    echo ""
    echo "📥 Please download it first:"
    echo "   Apple Silicon: https://www.oracle.com/database/technologies/instant-client/macos-arm64-downloads.html"
    echo "   Intel Mac: https://www.oracle.com/database/technologies/instant-client/macosx-x86-64-downloads.html"
    echo ""
    echo "   Download the 'Basic Package' ZIP file, then run this script again."
    exit 1
fi

echo "✅ Found: $(basename "$ZIP_FILE")"
echo "📦 Extracting to $INSTALL_DIR..."

# Create directory and extract
mkdir -p "$(dirname "$INSTALL_DIR")"
unzip -q "$ZIP_FILE" -d "$(dirname "$INSTALL_DIR")"

# Find the extracted directory (version number may vary)
EXTRACTED_DIR=$(find "$(dirname "$INSTALL_DIR")" -type d -name "instantclient_*" | head -1)

if [ -z "$EXTRACTED_DIR" ]; then
    echo "❌ Extraction failed or directory not found"
    exit 1
fi

echo "✅ Extracted to: $EXTRACTED_DIR"

# Add to .zshrc
if ! grep -q "DYLD_LIBRARY_PATH.*instantclient" ~/.zshrc 2>/dev/null; then
    echo "" >> ~/.zshrc
    echo "# Oracle Instant Client" >> ~/.zshrc
    echo "export DYLD_LIBRARY_PATH=$EXTRACTED_DIR:\$DYLD_LIBRARY_PATH" >> ~/.zshrc
    echo "✅ Added to ~/.zshrc"
else
    echo "⚠️  DYLD_LIBRARY_PATH already configured in ~/.zshrc"
fi

# Source it for current session
export DYLD_LIBRARY_PATH="$EXTRACTED_DIR:$DYLD_LIBRARY_PATH"

echo ""
echo "✅ Installation complete!"
echo ""
echo "📝 To use in Python:"
echo "   import oracledb"
echo "   oracledb.init_oracle_client(lib_dir=\"$EXTRACTED_DIR\")"
echo ""
echo "💡 Restart your terminal or run: source ~/.zshrc"
