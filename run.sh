#!/usr/bin/env bash
set -e

# Tracker Failure Simulator runner script
# Starts both the FastAPI backend and Next.js frontend concurrently.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
BACKEND_HOST="${BACKEND_HOST:-0.0.0.0}"

if [ "$1" = "down" ] || [ "$1" = "stop" ]; then
    exec "$SCRIPT_DIR/stop.sh"
fi

# Colors for terminal output
BOLD='\033[1m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BOLD}${CYAN}================================================${NC}"
echo -e "${BOLD}${CYAN}      Tracker Failure Simulator Runner          ${NC}"
echo -e "${BOLD}${CYAN}================================================${NC}"

# Helper function to release/kill any process occupying a specified TCP port
free_port() {
    local port="$1"
    local name="$2"
    local pids=""

    if command -v fuser &>/dev/null; then
        pids=$(fuser "${port}/tcp" 2>/dev/null || true)
    fi

    if [ -z "$pids" ] && command -v lsof &>/dev/null; then
        pids=$(lsof -ti "tcp:${port}" 2>/dev/null || true)
    fi

    if [ -z "$pids" ] && command -v ss &>/dev/null; then
        pids=$(ss -tlpn 2>/dev/null | grep -E ":${port}\b" | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u || true)
    fi

    if [ -n "$pids" ]; then
        echo -e "${YELLOW}[Port $port ($name)] Closing existing process occupying port (PID: $pids)...${NC}"
        # Graceful SIGTERM
        for pid in $pids; do
            if [ -n "$pid" ] && [ "$pid" -gt 1 ] && [ "$pid" -ne "$$" ]; then
                kill -15 "$pid" 2>/dev/null || true
            fi
        done
        sleep 0.5
        # Force SIGKILL if still running
        for pid in $pids; do
            if [ -n "$pid" ] && [ "$pid" -gt 1 ] && [ "$pid" -ne "$$" ] && kill -0 "$pid" 2>/dev/null; then
                kill -9 "$pid" 2>/dev/null || true
            fi
        done
        # Fallback to fuser -k
        if command -v fuser &>/dev/null; then
            fuser -k -9 "${port}/tcp" 2>/dev/null || true
        fi
        sleep 0.3
        echo -e "${GREEN}[Port $port ($name)] Port released successfully.${NC}"
    fi
}

# 1. Free ports if already occupied
free_port "$BACKEND_PORT" "Backend"
free_port "$FRONTEND_PORT" "Frontend"

# 2. Ensure backend environment
PYTHON_BIN="$BACKEND_DIR/.venv/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    echo -e "${YELLOW}[Backend] Python virtualenv not found in $BACKEND_DIR/.venv.${NC}"
    if command -v uv &>/dev/null; then
        echo -e "${CYAN}[Backend] Initializing venv with uv...${NC}"
        (cd "$BACKEND_DIR" && uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt)
    elif command -v python3 &>/dev/null; then
        echo -e "${CYAN}[Backend] Initializing venv with python3 -m venv...${NC}"
        (cd "$BACKEND_DIR" && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt)
    else
        echo -e "${RED}[Backend] Error: Neither uv nor python3 is installed.${NC}"
        exit 1
    fi
fi

if [ ! -f "$PYTHON_BIN" ]; then
    echo -e "${RED}[Backend] Error: Virtual environment python executable not found at $PYTHON_BIN${NC}"
    exit 1
fi

# 3. Ensure frontend dependencies
if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    echo -e "${YELLOW}[Frontend] node_modules not found in $FRONTEND_DIR. Running npm install...${NC}"
    (cd "$FRONTEND_DIR" && npm install)
fi

# Process management and clean teardown
BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM
    echo -e "\n${YELLOW}Stopping services...${NC}"
    for pid in "$BACKEND_PID" "$FRONTEND_PID"; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            pkill -P "$pid" 2>/dev/null || true
            kill "$pid" 2>/dev/null || true
        fi
    done
    wait "$BACKEND_PID" 2>/dev/null || true
    wait "$FRONTEND_PID" 2>/dev/null || true

    # Ensure ports are clean on exit
    if command -v fuser &>/dev/null; then
        fuser -k "${BACKEND_PORT}/tcp" 2>/dev/null || true
        fuser -k "${FRONTEND_PORT}/tcp" 2>/dev/null || true
    fi

    echo -e "${GREEN}All services stopped.${NC}"
    exit "$exit_code"
}

trap cleanup EXIT INT TERM

# Start Backend
echo -e "${GREEN}[Backend]  Starting uvicorn on http://${BACKEND_HOST}:${BACKEND_PORT}...${NC}"
(
    cd "$BACKEND_DIR"
    exec "$PYTHON_BIN" -m uvicorn app.main:app --host "$BACKEND_HOST" --port "$BACKEND_PORT" --reload
) &
BACKEND_PID=$!

# Start Frontend
echo -e "${GREEN}[Frontend] Starting Next.js on http://localhost:${FRONTEND_PORT}...${NC}"
(
    cd "$FRONTEND_DIR"
    export NEXT_PUBLIC_API_URL="${NEXT_PUBLIC_API_URL:-http://localhost:${BACKEND_PORT}}"
    exec npx next dev -p "$FRONTEND_PORT"
) &
FRONTEND_PID=$!

echo -e "\n${BOLD}${GREEN}Ready!${NC}"
echo -e "  • Frontend: ${CYAN}http://localhost:${FRONTEND_PORT}${NC}"
echo -e "  • Backend:  ${CYAN}http://localhost:${BACKEND_PORT}${NC} (Docs: http://localhost:${BACKEND_PORT}/docs)"
echo -e "${YELLOW}Press Ctrl+C to stop both servers.${NC}\n"

# Wait for either process to terminate
wait -n "$BACKEND_PID" "$FRONTEND_PID" || true
