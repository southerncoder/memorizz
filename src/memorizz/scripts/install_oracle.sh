#!/usr/bin/env bash
# Oracle Database Installation Script for MemoRizz
#
# This script installs and starts Oracle Database 23ai Free in a Docker container.
# It checks if a container already exists and starts it, or creates a new one.
#
# Usage:
#   ./install_oracle.sh
#
#   Or with custom password:
#   export ORACLE_ADMIN_PASSWORD="YourSecurePassword123!"
#   ./install_oracle.sh
#
#   For Apple Silicon (M1/M2/M3):
#   export PLATFORM_FLAG="--platform linux/amd64"
#   ./install_oracle.sh
#
#   For Oracle image version:
#   export ORACLE_IMAGE_TAG="latest-lite"  # Lite version (1.78GB, default)
#   export ORACLE_IMAGE_TAG="latest"        # Full version (9.93GB)
#   export ORACLE_IMAGE_TAG="custom-tag"   # Any custom tag
#   ./install_oracle.sh
#
# Environment Variables:
#   ORACLE_ADMIN_PASSWORD  - Admin password (default: MyPassword123!)
#   ORACLE_IMAGE_TAG       - Oracle image tag (default: latest-lite)
#                            Options: latest-lite (1.78GB), latest (9.93GB), or custom tag
#   PLATFORM_FLAG          - Docker platform flag (default: empty, auto-detect)
#                            Use "--platform linux/amd64" for Apple Silicon
#
# Features:
#   - Persistent data storage via Docker volume (oracle-memorizz-data)
#   - Idempotent: safe to run multiple times
#   - Waits for database readiness before completing
#   - Cross-platform support (Intel, AMD, Apple Silicon)
#
# See SETUP.md for complete setup instructions.

set -e  # Exit immediately on error

CONTAINER_NAME="oracle-memorizz"
VOLUME_NAME="oracle-memorizz-data"

# Oracle image tag - defaults to latest-lite (smaller, faster download)
# Options: latest-lite (1.78GB, default), latest (9.93GB), or any custom tag
ORACLE_IMAGE_TAG="${ORACLE_IMAGE_TAG:-latest-lite}"
IMAGE_NAME="container-registry.oracle.com/database/free:${ORACLE_IMAGE_TAG}"

# Use environment variable if set, otherwise use default
# Set ORACLE_ADMIN_PASSWORD to customize the admin password
PASSWORD="${ORACLE_ADMIN_PASSWORD:-MyPassword123!}"

# Platform flag for Apple Silicon compatibility
# Oracle Free may require emulation on ARM64
PLATFORM_FLAG="${PLATFORM_FLAG:-}"

# --- Helper function ---
function log() {
  echo -e "\033[1;36m$1\033[0m" >&2
}

function error() {
  echo -e "\033[1;31m$1\033[0m" >&2
}

# Check if Docker is running
log "🔍 Checking if Docker is running..."
if ! docker info >/dev/null 2>&1; then
  error ""
  error "❌ Docker is not running!"
  error ""
  error "Please start Docker Desktop (or Docker daemon) and try again."
  error ""
  error "To verify Docker is running:"
  error "  docker ps"
  error ""
  exit 1
fi

log "✅ Docker is running"
log "📦 Using Oracle image tag: ${ORACLE_IMAGE_TAG}"
log "🔍 Checking if Oracle container '$CONTAINER_NAME' exists..."

# Check if container exists
if [ "$(docker ps -a -q -f name=$CONTAINER_NAME)" ]; then
  # Check if container is already running
  if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    log "✅ Container '$CONTAINER_NAME' is already running!"
  else
    log "▶️ Starting existing container '$CONTAINER_NAME'..."
    docker start $CONTAINER_NAME
  fi
else
  # Check if image already exists locally
  log "🔍 Checking if Oracle image exists locally..."
  IMAGE_EXISTS=$(docker images -q $IMAGE_NAME 2>/dev/null)

  if [ -n "$IMAGE_EXISTS" ]; then
    log "✅ Oracle image found locally, skipping download"
  else
    if [ "$ORACLE_IMAGE_TAG" = "latest-lite" ]; then
      log "🐳 Pulling Oracle Database 23ai Free Lite (1.78GB, with AI Vector Search)..."
    elif [ "$ORACLE_IMAGE_TAG" = "latest" ]; then
      log "🐳 Pulling Oracle Database 23ai Free Full (9.93GB, with AI Vector Search)..."
    else
      log "🐳 Pulling Oracle Database 23ai Free (tag: ${ORACLE_IMAGE_TAG})..."
    fi
    if [ -n "$PLATFORM_FLAG" ]; then
      log "   Using platform flag: $PLATFORM_FLAG (for Apple Silicon compatibility)"
      docker pull $PLATFORM_FLAG $IMAGE_NAME
    else
      docker pull $IMAGE_NAME
    fi
  fi

  log "📦 Creating persistent volume '$VOLUME_NAME' (if not exists)..."
  docker volume create $VOLUME_NAME 2>/dev/null || log "   Volume already exists (reusing)"

  log "🚀 Creating and starting new container '$CONTAINER_NAME'..."
  log "   Using persistent volume: $VOLUME_NAME"
  if [ -n "$PLATFORM_FLAG" ]; then
    docker run -d \
      $PLATFORM_FLAG \
      --name $CONTAINER_NAME \
      -p 1521:1521 \
      -e ORACLE_PWD=$PASSWORD \
      -v $VOLUME_NAME:/opt/oracle/oradata \
      $IMAGE_NAME
  else
    docker run -d \
      --name $CONTAINER_NAME \
      -p 1521:1521 \
      -e ORACLE_PWD=$PASSWORD \
      -v $VOLUME_NAME:/opt/oracle/oradata \
      $IMAGE_NAME
  fi
fi

log "⏳ Waiting for Oracle to be ready (this may take 2–3 minutes)..."
log "   Monitoring container logs for readiness..."
# Wait until the container logs show "DATABASE IS READY TO USE!"
# Using 5-second intervals for more responsive feedback
until docker logs $CONTAINER_NAME 2>&1 | grep -q "DATABASE IS READY TO USE!"; do
  sleep 5
  echo -n "." >&2
done

echo "" >&2
log "✅ Oracle Database is ready!"
echo "" >&2
echo "Connection details:" >&2
echo "  Host: localhost" >&2
echo "  Port: 1521" >&2
echo "  Service Name: FREEPDB1" >&2
echo "  Admin User: system" >&2
echo "  Admin Password: $PASSWORD" >&2
echo "" >&2
echo "📝 Environment Variables:" >&2
echo "  To use these credentials in your shell, run:" >&2
echo "    eval \$(./install_oracle.sh)" >&2
echo "" >&2
echo "  Or source the script:" >&2
echo "    source ./install_oracle.sh" >&2
echo "" >&2
echo "  Or set manually:" >&2
echo "    export ORACLE_ADMIN_PASSWORD=\"$PASSWORD\"" >&2
echo "    export ORACLE_USER=\"memorizz_user\"" >&2
echo "    export ORACLE_PASSWORD=\"SecurePass123!\"" >&2
echo "    export ORACLE_DSN=\"localhost:1521/FREEPDB1\"" >&2
echo "" >&2
# Export variables (useful if script is sourced)
export ORACLE_ADMIN_PASSWORD="$PASSWORD"
export ORACLE_USER="memorizz_user"
export ORACLE_PASSWORD="SecurePass123!"
export ORACLE_DSN="localhost:1521/FREEPDB1"
# Output export commands to stdout (for eval) - these must be clean, no colors
echo "export ORACLE_ADMIN_PASSWORD=\"$PASSWORD\""
echo "export ORACLE_USER=\"memorizz_user\""
echo "export ORACLE_PASSWORD=\"SecurePass123!\""
echo "export ORACLE_DSN=\"localhost:1521/FREEPDB1\""
