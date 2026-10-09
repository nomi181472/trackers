#!/usr/bin/env bash
set -e

# Tracker Failure Simulator shutdown script
# Terminates all running backend and frontend processes and frees ports 8000 and 3000.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"

# Colors
BOLD='\033[1m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BOLD}${CYAN}================================================${NC}"
echo -e "${BOLD}${CYAN}      Tracker Failure Simulator Teardown        ${NC}"
echo -e "${BOLD}${CYAN}================================================${NC}"

stopped_any=false

# 0. Check and stop Docker Compose containers if running
stop_docker() {
    if [ -f "$SCRIPT_DIR/docker-compose.yml" ]; then
        if command -v docker &>/dev/null && docker compose version &>/dev/null; then
            if docker compose -f "$SCRIPT_DIR/docker-compose.yml" ps --status running -q 2>/dev/null | grep -q .; then
                stopped_any=true
                echo -e "${YELLOW}[Docker] Stopping running docker compose services...${NC}"
                docker compose -f "$SCRIPT_DIR/docker-compose.yml" down 2>/dev/null || true
                echo -e "${GREEN}[Docker] Docker Compose services stopped.${NC}"
            fi
        elif command -v docker-compose &>/dev/null; then
            if docker-compose -f "$SCRIPT_DIR/docker-compose.yml" ps -q 2>/dev/null | grep -q .; then
                stopped_any=true
                echo -e "${YELLOW}[Docker] Stopping running docker-compose services...${NC}"
                docker-compose -f "$SCRIPT_DIR/docker-compose.yml" down 2>/dev/null || true
                echo -e "${GREEN}[Docker] Docker Compose services stopped.${NC}"
            fi
        fi
    fi
}

if [ "$1" = "docker" ] || [ "$1" = "compose" ]; then
    stop_docker
    echo -e "\n${BOLD}${GREEN}✔ Docker services have been stopped.${NC}\n"
    exit 0
fi

stop_docker

# Helper to find and kill processes on a specific TCP port
stop_port() {
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
        stopped_any=true
        echo -e "${YELLOW}[$name] Stopping processes on port $port (PID(s): $pids)...${NC}"
        for pid in $pids; do
            if [ -n "$pid" ] && [ "$pid" -gt 1 ] && [ "$pid" -ne "$$" ]; then
                pkill -P "$pid" 2>/dev/null || true
                kill -15 "$pid" 2>/dev/null || true
            fi
        done

        sleep 0.6

        # Force kill if still lingering
        for pid in $pids; do
            if [ -n "$pid" ] && [ "$pid" -gt 1 ] && [ "$pid" -ne "$$" ] && kill -0 "$pid" 2>/dev/null; then
                kill -9 "$pid" 2>/dev/null || true
            fi
        done

        if command -v fuser &>/dev/null; then
            fuser -k -9 "${port}/tcp" 2>/dev/null || true
        fi
        sleep 0.2
        echo -e "${GREEN}[$name] Port $port freed.${NC}"
    else
        echo -e "${CYAN}[$name] No active process found on port $port.${NC}"
    fi
}

# Also kill by command pattern as backup (in case processes spawned without binding yet)
stop_patterns() {
    # Backend uvicorn workers
    local uvicorn_pids
    uvicorn_pids=$(pgrep -f "uvicorn.*app.main:app" || true)
    if [ -n "$uvicorn_pids" ]; then
        stopped_any=true
        echo -e "${YELLOW}[Backend] Stopping lingering uvicorn processes (PID(s): $uvicorn_pids)...${NC}"
        for pid in $uvicorn_pids; do
            kill -15 "$pid" 2>/dev/null || true
        done
    fi

    # Next.js dev server processes
    local next_pids
    next_pids=$(pgrep -f "next-server.*3000|next dev" || true)
    if [ -n "$next_pids" ]; then
        stopped_any=true
        echo -e "${YELLOW}[Frontend] Stopping lingering Next.js processes (PID(s): $next_pids)...${NC}"
        for pid in $next_pids; do
            kill -15 "$pid" 2>/dev/null || true
        done
    fi
}

stop_port "$BACKEND_PORT" "Backend"
stop_port "$FRONTEND_PORT" "Frontend"
stop_patterns

echo -e "\n${BOLD}${GREEN}✔ All services (Backend & Frontend) have been stopped.${NC}\n"
