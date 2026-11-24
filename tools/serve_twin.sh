#!/bin/bash
# Serve a saved twin (operator page + REST API) as it was at the end of its last flight, until Ctrl+C.
# View only: no new flight, no new data, nothing is written, so it is safe on result files.
# usage: bash tools/serve_twin.sh STATE.json [PORT]        (default port 8080)
#   then open http://localhost:PORT  (page)  and  http://localhost:PORT/api/v1  (REST API)
#   Several viewers (different ports) and a running flight can be used at the same time.

WS=$HOME/ROS/thesis_ws
STATE=$1
PORT=${2:-8080}
[ -n "$STATE" ] || { sed -n 2,5p "$0"; exit 2; }
case $STATE in /*) ;; *) STATE=$PWD/$STATE ;; esac
[ -f "$STATE" ] || { echo "state file not found: $STATE"; exit 1; }
if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
  echo "port $PORT is already in use (a flight or another viewer?). Use another port, e.g.: $0 $1 8090"; exit 1
fi
# Each viewer gets its own ROS domain, so it never exchanges data with a flight or another viewer
# (flights use the default domain 0).
export ROS_DOMAIN_ID=$(( 51 + PORT % 50 ))

source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"
LOG=$(mktemp -d)
PIDS=()
stop() {
  for p in "${PIDS[@]}"; do kill -INT -- -"$p" 2>/dev/null; done
  sleep 2
  for p in "${PIDS[@]}"; do kill -9 -- -"$p" 2>/dev/null; done
  pkill -9 -f "[t]win_node.*view_only:=true.*$STATE" 2>/dev/null
  pkill -9 -f "[o]perator_node.*port:=$PORT" 2>/dev/null
  echo "stopped"
}
trap stop EXIT
trap 'exit 0' INT TERM

setsid --fork ros2 run semantic_twin twin_node --ros-args -p load_state:=true -p view_only:=true \
  -p state_path:="$STATE" > "$LOG/twin.log" 2>&1 &
sleep 1; TWPID=$(pgrep -n -f "[t]win_node.*view_only:=true"); PIDS+=($(ps -o pgid= -p "$TWPID" | tr -d ' '))
setsid --fork ros2 run semantic_twin operator_node --ros-args -p port:="$PORT" > "$LOG/operator.log" 2>&1 &
sleep 1; OPPID=$(pgrep -n -f "[o]perator_node.*port:=$PORT"); PIDS+=($(ps -o pgid= -p "$OPPID" | tr -d ' '))

for i in $(seq 30); do grep -q "Twin node ready" "$LOG/twin.log" && break; sleep 1; done
grep -q "Twin node ready" "$LOG/twin.log" || { echo "twin node did not start:"; tail -5 "$LOG/twin.log"; exit 1; }
echo "$(grep -o 'Loaded twin.*' "$LOG/twin.log")"
echo "$(grep -o 'current flight.*' "$LOG/twin.log")"
echo
echo "  operator page: http://localhost:$PORT"
echo "  REST API:      http://localhost:$PORT/api/v1"
echo
echo "Press Ctrl+C to stop."
while kill -0 "$TWPID" 2>/dev/null && kill -0 "$OPPID" 2>/dev/null; do sleep 1; done
echo "a node stopped unexpectedly:"; tail -3 "$LOG/twin.log" "$LOG/operator.log"
