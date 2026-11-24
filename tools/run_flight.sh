#!/bin/bash
# One flight, start to finish, in one terminal:
#   checks -> simulation + PX4 + MAVROS + mission -> the 5 nodes (each once) -> checks -> take-off
#   -> waits for the end of the route (stops if a node dies) -> exports -> saves -> stops everything -> scores.
#
# usage:
#   bash tools/run_flight.sh --name flight_7 --world disaster_scene_changed.world \
#        --state results/series2/twin_state.json --load --out results/flight7_changed \
#        [--before-world disaster_scene.world] [--route inspection|survey] [--gui]
#
#   --new   start a new, empty twin (refuses if --state already exists)
#   --load  continue the twin in --state (refuses if it does not exist); a copy is kept as OUT/twin_state_before.json
#   --before-world   also score the change report (evaluate_changes.py) against this "before" world
#   --route  inspection (8 m, default) or survey (12 m)
#   --serve  after the scores, keep the operator page and REST API running for this twin (until Ctrl+C)
#   --no-detector  do not start cv_node (mapping only, e.g. the 12 m survey); the twin gets no people from this flight
# Exports, logs and the final twin state all go to --out. /tmp is not used. results/twin_state.json is never touched.

WS=$HOME/ROS/thesis_ws
W=$WS/src/semantic_twin/worlds
NAME= WORLD= STATE= MODE= OUT= BEFORE= ROUTE=inspection GUI=gui:=false DETECT=1 SERVE=0
while [ $# -gt 0 ]; do
  case $1 in
    --name) NAME=$2; shift ;;
    --world) WORLD=$2; shift ;;
    --state) STATE=$2; shift ;;
    --new) MODE=new ;;
    --load) MODE=load ;;
    --out) OUT=$2; shift ;;
    --before-world) BEFORE=$2; shift ;;
    --route) ROUTE=$2; shift ;;
    --gui) GUI=gui:=true ;;
    --no-detector) DETECT=0 ;;
    --serve) SERVE=1 ;;
    *) echo "unknown argument: $1"; exit 2 ;;
  esac
  shift
done

say()  { echo "[$(date +%T)] $*"; }
fail() { echo; echo "[$(date +%T)] STOPPED: $*"; exit 1; }

# ---------- argument checks
[ -n "$NAME" ] && [ -n "$WORLD" ] && [ -n "$STATE" ] && [ -n "$MODE" ] && [ -n "$OUT" ] \
  || { sed -n 2,17p "$0"; exit 2; }
cd "$WS" || exit 1
case $STATE in /*) ;; *) STATE=$WS/$STATE ;; esac
case $OUT in /*) ;; *) OUT=$WS/$OUT ;; esac
[ -f "$W/$WORLD" ] || fail "world not found: $W/$WORLD"
[ -z "$BEFORE" ] || [ -f "$W/$BEFORE" ] || fail "before-world not found: $W/$BEFORE"
case $ROUTE in inspection|survey) ;; *) fail "--route must be inspection or survey" ;; esac
[ "$(realpath -m "$STATE")" = "$(realpath -m "$WS/results/twin_state.json")" ] && fail "refusing to fly on results/twin_state.json (use a copy)"
if [ "$MODE" = new ]; then
  [ -e "$STATE" ] && fail "--new, but $STATE already exists (move it away first)"
else
  [ -f "$STATE" ] || fail "--load, but $STATE does not exist"
fi
mkdir -p "$OUT/logs" "$(dirname "$STATE")"
ls "$OUT"/sdt_* >/dev/null 2>&1 && fail "$OUT already contains results (sdt_* files); use an empty folder"
[ -f "$WS/yolov8s.pt" ] || fail "yolov8s.pt missing in $WS"

source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"
[ -f "$HOME/ROS/px4_gazebo_env.sh" ] && source "$HOME/ROS/px4_gazebo_env.sh"
export PYTHONUNBUFFERED=1

# ---------- nothing from an earlier run may be alive
LEFT='[m]ission_node|[m]avros_node|[g]zserver|[g]zclient|bin/[p]x4|semantic_twin/[l]ib/'
# flight processes only: viewers started by serve_twin.sh (view-only twin node, operator node with its own
# port) run in their own ROS domain and are neither counted nor stopped here
left_pids() { pgrep -fa "$LEFT" | grep -v -e 'view_only:=true' -e 'operator_node.*port:=' | awk '{print $1}'; }
left_list() { pgrep -fa "$LEFT" | grep -v -e 'view_only:=true' -e 'operator_node.*port:=' | cut -c1-120; }
if [ -n "$(left_pids)" ]; then
  left_list
  fail "processes from an earlier run are still running (listed above). Close them first."
fi
if ss -ltn 2>/dev/null | grep -q ":8080 "; then
  fail "port 8080 is in use (a viewer from serve_twin.sh?). Stop it, or start viewers on another port (e.g. 8090)."
fi
ros2 daemon stop >/dev/null 2>&1

# ---------- start/stop helpers: every program in its own process group
PIDS=()
declare -A NODE_PID
start() {   # start LOGNAME command...
  local log=$OUT/logs/$1.log; shift
  setsid "$@" > "$log" 2>&1 &
  PIDS+=($!)
  LAST_PID=$!
}
STOPPED=0
stop_all() {
  [ $STOPPED = 1 ] && return; STOPPED=1
  say "stopping everything ..."
  for p in "${PIDS[@]}"; do kill -INT -- -"$p" 2>/dev/null; done
  for i in $(seq 15); do [ -z "$(left_pids)" ] && break; sleep 1; done
  for p in "${PIDS[@]}"; do kill -9 -- -"$p" 2>/dev/null; done
  sleep 1
  for p in $(left_pids); do kill -9 "$p" 2>/dev/null; done
  sleep 1
  if [ -n "$(left_pids)" ]; then say "WARNING: some processes are still running:"; left_list
  else say "all stopped"; fi
}
trap 'stop_all' EXIT
trap 'echo; fail "interrupted (Ctrl+C)"' INT TERM

wait_for() {   # wait_for FILE PATTERN SECONDS
  for i in $(seq "$3"); do grep -q -- "$2" "$1" 2>/dev/null && return 0; sleep 1; done
  return 1
}
alive() { kill -0 "$1" 2>/dev/null; }

say "flight $NAME | world $WORLD | route $ROUTE | detector $([ $DETECT = 1 ] && echo on || echo OFF) | twin $MODE: $STATE"
say "results -> $OUT"
[ "$MODE" = load ] && cp "$STATE" "$OUT/twin_state_before.json" && say "copy of the twin before this flight: $OUT/twin_state_before.json"

# ---------- 1. simulation, PX4, MAVROS, mission
say "1/7 starting simulation, PX4, MAVROS and mission ($ROUTE route)"
start launch bash "$WS/tools/fly_$ROUTE.sh" "$GUI" "$WORLD"
LAUNCH_PID=$LAST_PID
wait_for "$OUT/logs/launch.log" "Waiting for start" 180 || fail "mission node not ready after 180 s (see $OUT/logs/launch.log)"
wait_for "$OUT/logs/launch.log" "Got HEARTBEAT" 60 || fail "MAVROS got no heartbeat from PX4 (see launch.log)"
sleep 10
RATE=$(timeout 10 ros2 topic hz /uav0/mavros/local_position/pose 2>/dev/null | grep -o "average rate: [0-9.]*" | tail -1 | awk '{print $3}')
say "pose rate: ${RATE:-none} Hz"
[ -n "$RATE" ] && awk "BEGIN{exit !($RATE > 20)}" || fail "pose rate too low or missing (expected about 30 Hz)"

# ---------- 2. the five nodes, each once
say "2/7 starting the nodes"
P="--ros-args -p use_sim_time:=true"
if [ $DETECT = 1 ]; then
  start cv ros2 run semantic_twin cv_node $P -p model_path:=$WS/yolov8s.pt -p imgsz:=960 -p confidence_threshold:=0.5
  NODE_PID[cv_node]=$LAST_PID
else
  say "detector OFF (--no-detector): mapping only"
fi
start integration ros2 run semantic_twin integration_node $P \
  -p log_path:=$OUT/sdt_detections.csv -p export_path:=$OUT/sdt_semantic_map.json
NODE_PID[integration_node]=$LAST_PID
start map ros2 run semantic_twin map_node $P -p export_path:=$OUT/sdt_height_map.json
NODE_PID[map_node]=$LAST_PID
TW="-p state_path:=$STATE -p export_path:=$OUT/sdt_twin.json -p flight_name:=$NAME"
[ "$MODE" = load ] && TW="$TW -p load_state:=true"
start twin ros2 run semantic_twin twin_node $P $TW
NODE_PID[twin_node]=$LAST_PID
start operator ros2 run semantic_twin operator_node $P
NODE_PID[operator_node]=$LAST_PID

# ---------- 3. checks before take-off
say "3/7 checks before take-off"
wait_for "$OUT/logs/twin.log" "current flight: $NAME" 60 || fail "twin node did not start flight $NAME (see logs/twin.log)"
if [ "$MODE" = load ]; then
  grep -q "Loaded twin from $STATE" "$OUT/logs/twin.log" || fail "twin node did not load $STATE"
  say "twin: $(grep -o 'Loaded twin.*' "$OUT/logs/twin.log" | sed 's/from .*://')"
else
  grep -q "Loaded twin" "$OUT/logs/twin.log" && fail "--new, but the twin node loaded a state"
  say "twin: new, empty"
fi
wait_for "$OUT/logs/map.log" "waited=" 40 || fail "map node gives no status with 'waited=' (old code? run colcon build)"
sleep 15
for n in "${!NODE_PID[@]}"; do alive "${NODE_PID[$n]}" || fail "$n died during start (see $OUT/logs/)"; done
NODES="/sdt_integration_node /sdt_map_node /sdt_twin_node /sdt_operator_node /uav0/mission"
[ $DETECT = 1 ] && NODES="/cv_node $NODES"
# ROS 2 discovery can take a few seconds on a busy machine: retry before deciding
for try in 1 2 3 4 5 6; do
  NL=$(ros2 node list 2>/dev/null); BAD=
  for n in $NODES; do
    c=$(echo "$NL" | grep -cx "$n")
    [ "$c" = 1 ] || BAD="$BAD $n:$c"
  done
  [ -z "$BAD" ] && break
  say "  node check $try/6 not complete yet (name:count):$BAD"
  sleep 5
done
[ -z "$BAD" ] || fail "nodes not running exactly once after 6 checks (name:count):$BAD"
[ $DETECT = 0 ] && [ "$(echo "$NL" | grep -cx /cv_node)" != 0 ] && fail "--no-detector, but a cv_node is running"
say "all $(echo $NODES | wc -w) nodes run exactly once"
curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/ | grep -q 200 && say "operator page: http://localhost:8080"

# ---------- 4. take-off
say "4/7 take-off"
R=$(timeout 30 ros2 service call /uav0/start_mission std_srvs/srv/Trigger 2>&1)
echo "$R" | grep -q "success=True" || fail "take-off trigger failed: $(echo "$R" | tail -1)"
say "mission started"

# ---------- 5. fly the route, watching the nodes
say "5/7 flying (status every 30 s)"
MAXT=$([ "$ROUTE" = survey ] && echo 3600 || echo 1200)
T0=$(date +%s); LASTSHOW=0
while ! grep -q -- "-> DONE" "$OUT/logs/launch.log"; do
  sleep 2
  alive "$LAUNCH_PID" || fail "simulation/PX4/mission stopped during the flight (see logs/launch.log)"
  for n in "${!NODE_PID[@]}"; do
    alive "${NODE_PID[$n]}" || fail "$n stopped during the flight. The twin state now contains part of this flight; restore it from $OUT/twin_state_before.json (if --load) before flying again"
  done
  now=$(date +%s)
  [ $((now - T0)) -gt $MAXT ] && fail "route not finished after $MAXT s"
  if [ $((now - LASTSHOW)) -ge 30 ]; then
    LASTSHOW=$now
    wp=$(grep -c "WP .* reached" "$OUT/logs/launch.log")
    m=$(grep -o "used=[0-9]* no_pose=[0-9]*" "$OUT/logs/map.log" | tail -1)
    t=$(grep -o "twin: .*structures" "$OUT/logs/twin.log" | tail -1)
    say "  waypoints reached: $wp | map: $m | $t"
  fi
done
say "route finished (-> DONE)"
sleep 6

# ---------- 6. export and save
say "6/7 exporting"
for s in /sdt_integration_node/export_map /sdt_map_node/export_grid /sdt_twin_node/export_twin; do
  R=$(timeout 60 ros2 service call $s std_srvs/srv/Trigger 2>&1)
  echo "$R" | grep -q "success=True" || fail "export $s failed: $(echo "$R" | tail -1)"
  say "  $(echo "$R" | grep -o "message=.*" | cut -c10-220)"
done
for f in sdt_semantic_map.json sdt_height_map.json sdt_twin.json; do
  python3 -I -c "import json,sys; json.load(open(sys.argv[1]))" "$OUT/$f" || fail "$f is not valid JSON"
done
cp "$STATE" "$OUT/twin_state.json"
say "saved in $OUT"

stop_all

# ---------- 7. scores
say "7/7 scores"
[ $DETECT = 1 ] && { python3 -I tools/evaluate_map.py "$OUT/sdt_semantic_map.json" "$W/$WORLD" | tee "$OUT/evaluate_map.txt"; echo; }
python3 -I tools/evaluate_grid.py "$OUT/sdt_height_map.json" "$W/$WORLD" | tee "$OUT/evaluate_grid.txt"
if [ -n "$BEFORE" ]; then
  echo
  python3 -I tools/evaluate_changes.py "$OUT/sdt_twin.json" "$W/$BEFORE" "$W/$WORLD" | tee "$OUT/evaluate_changes.txt"
fi
echo
say "done. Results and logs: $OUT"
if [ $SERVE = 1 ]; then
  echo
  exec bash "$WS/tools/serve_twin.sh" "$OUT/twin_state.json"
fi
echo "To look at this twin again (page + API): bash tools/serve_twin.sh $OUT/twin_state.json"
