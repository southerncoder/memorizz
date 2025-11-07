#!/usr/bin/env bash
set -e  # Exit immediately on error

CONTAINER_NAME="oracle-memorizz"
IMAGE_NAME="container-registry.oracle.com/database/free:latest"
PASSWORD="MyPassword123!"

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
  docker pull $IMAGE_NAME

  log "🚀 Creating and starting new container '$CONTAINER_NAME'..."
  docker run -d \
    --name $CONTAINER_NAME \
    -p 1521:1521 \
    -e ORACLE_PWD=$PASSWORD \
    $IMAGE_NAME
fi

log "⏳ Waiting for Oracle to be ready (this may take 2–3 minutes)..."
# Wait until the container logs show "DATABASE IS READY TO USE!"
until docker logs $CONTAINER_NAME 2>&1 | grep -q "DATABASE IS READY TO USE!"; do
  sleep 10
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
