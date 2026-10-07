#!/bin/bash
set -e

echo "Starting FastAPI backend on port 8000..."
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --app-dir /app/backend &
BACKEND_PID=$!

echo "Starting Next.js frontend on port 3000..."
PORT=3000 node /app/frontend/server.js &
FRONTEND_PID=$!

echo "Starting Reverse Proxy Gateway on port 7860..."
node /app/gateway.js &
GATEWAY_PID=$!

# Trap signals for graceful shutdown
trap "kill -TERM $BACKEND_PID $FRONTEND_PID $GATEWAY_PID 2>/dev/null || true; exit 0" SIGINT SIGTERM EXIT

# Keep alive and exit if any process terminates
wait -n $BACKEND_PID $FRONTEND_PID $GATEWAY_PID
