#!/usr/bin/env bash
# Oracle Database Startup Script for MemoRizz
#
# This script starts Oracle Database 23ai Free in a Docker container.
# It checks if a container already exists and starts it, or creates a new one.
#
# Usage:
#   ./start_oracle.sh
#
#   Or with custom password:
#   export ORACLE_ADMIN_PASSWORD="YourSecurePassword123!"
#   ./start_oracle.sh
#
#   For Apple Silicon (M1/M2/M3):
#   export PLATFORM_FLAG="--platform linux/amd64"
#   ./start_oracle.sh
#
# Environment Variables:
#   ORACLE_ADMIN_PASSWORD  - Admin password (default: MyPassword123!)
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
IMAGE_NAME="container-registry.oracle.com/database/free:latest"

# Use environment variable if set, otherwise use default
# Set ORACLE_ADMIN_PASSWORD to customize the admin password
PASSWORD="${ORACLE_ADMIN_PASSWORD:-MyPassword123!}"

# Platform flag for Apple Silicon compatibility
# Oracle Free may require emulation on ARM64
PLATFORM_FLAG="${PLATFORM_FLAG:-}"

# --- Helper function ---
function log() {
  echo -e "\033[1;36m$1\033[0m"
}

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
  log "🐳 Pulling Oracle Database 23ai Free (with AI Vector Search)..."
  if [ -n "$PLATFORM_FLAG" ]; then
    log "   Using platform flag: $PLATFORM_FLAG (for Apple Silicon compatibility)"
    docker pull $PLATFORM_FLAG $IMAGE_NAME
  else
    docker pull $IMAGE_NAME
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
  echo -n "."
done

echo ""
log "✅ Oracle Database is ready!"
echo ""
echo "Connection details:"
echo "  Host: localhost"
echo "  Port: 1521"
echo "  Service Name: FREEPDB1"
echo "  Admin User: system"
echo "  Admin Password: $PASSWORD"
