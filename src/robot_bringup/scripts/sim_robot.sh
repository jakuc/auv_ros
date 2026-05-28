#!/bin/bash
# Uruchamia Isaac Sim + węzły ROS2 dla AUV
set -e

WORKSPACE=/workspace
VERBOSE=0
[[ "$1" == "--verbose" || "$1" == "-v" ]] && VERBOSE=1

cd "$WORKSPACE"
# shellcheck disable=SC1091
source "$WORKSPACE/install/setup.bash"

# xacro → URDF (tu ROS2 jest w PATH, w przeciwieństwie do środowiska Isaac Sim)
XACRO_PATH="$(ros2 pkg prefix bluerov2_description)/share/bluerov2_description/urdf/bluerov2.xacro"
URDF_PATH="/tmp/bluerov2.urdf"
echo "[sim_robot] xacro → URDF..."
ros2 run xacro xacro "$XACRO_PATH" > "$URDF_PATH"

# Isaac Sim nie rozwiązuje package:// URI — zamieniamy na ścieżki absolutne
DESCRIPTION_SHARE="$(ros2 pkg prefix bluerov2_description)/share/bluerov2_description"
sed -i "s|package://bluerov2_description|${DESCRIPTION_SHARE}|g" "$URDF_PATH"
echo "[sim_robot] URDF zapisany: $URDF_PATH (package:// → ścieżki absolutne)"

ISAAC_SCRIPT="$(ros2 pkg prefix robot_bringup)/share/robot_bringup/isaac/isaac_sim.py"

_cleanup() {
    echo "[sim_robot] Zamykam wszystkie procesy..."
    # kill -- "-PID" zabija całą grupę procesów (setsid → PID == PGID)
    [ -n "$ISAAC_PID"   ] && kill -- "-$ISAAC_PID"   2>/dev/null || kill "$ISAAC_PID"   2>/dev/null
    [ -n "$BRINGUP_PID" ] && kill -- "-$BRINGUP_PID" 2>/dev/null || kill "$BRINGUP_PID" 2>/dev/null
    [ -n "$NAVI_PID"    ] && kill -- "-$NAVI_PID"    2>/dev/null || kill "$NAVI_PID"    2>/dev/null
    wait 2>/dev/null
}
trap _cleanup EXIT INT TERM

# setsid: uruchamia python3 jako lidera nowej grupy procesów.
# Dzięki temu kill -- -$ISAAC_PID zabija python3 + wszystkie dzieci Isaac Sim.
# Kopiuj konfig RTX LiDAR do katalogu pluginu Isaac Sim (potrzebne po odtworzeniu kontenera)
LIDAR_CFG_SRC="$(ros2 pkg prefix robot_bringup)/share/robot_bringup/config/lidar_spherical.json"
LIDAR_CFG_DST="$HOME/.local/share/ov/data/exts/v2/omni.sensors.nv.common-2.5.0-coreapi+lx64.r.cp310/data/lidar/lidar_spherical.json"
[ -f "$LIDAR_CFG_SRC" ] && cp "$LIDAR_CFG_SRC" "$LIDAR_CFG_DST" && echo "[sim_robot] LiDAR config → $LIDAR_CFG_DST"

if [ "$VERBOSE" -eq 1 ]; then
    setsid python3 "$ISAAC_SCRIPT" &
else
    # W trybie pipe $! to PID grepa, nie python3 — używamy pgrep po nazwie skryptu
    setsid python3 "$ISAAC_SCRIPT" 2>&1 | grep -v "^\[Warning\]\|^Kit\|^Shader\|^\[omni" &
fi

# Krótkie oczekiwanie żeby python3 zdążył wystartować przed pgrep
sleep 0.3
ISAAC_PID=$(pgrep -f "python3.*isaac_sim.py" | head -1)
echo "[sim_robot] Isaac Sim PID: $ISAAC_PID"

sleep 5

setsid ros2 launch robot_bringup isaac.launch.py &
BRINGUP_PID=$!

setsid ros2 launch navi navi.launch.py &
NAVI_PID=$!

wait "$ISAAC_PID"
