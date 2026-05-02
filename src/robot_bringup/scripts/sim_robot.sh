#!/bin/bash
# Uruchamia Isaac Sim + węzły ROS2 dla AUV
set -e

WORKSPACE=/workspace
VERBOSE=0
[[ "$1" == "--verbose" || "$1" == "-v" ]] && VERBOSE=1

cd "$WORKSPACE"

# xacro → URDF (tu ROS2 jest w PATH, w przeciwieństwie do środowiska Isaac Sim)
XACRO_PATH="$(ros2 pkg prefix bluerov2_description)/share/bluerov2_description/urdf/bluerov2.xacro"
URDF_PATH="/tmp/bluerov2.urdf"
echo "[sim_robot] xacro → URDF..."
ros2 run xacro xacro "$XACRO_PATH" > "$URDF_PATH"

# Isaac Sim nie rozwiązuje package:// URI — zamieniamy na ścieżki absolutne
DESCRIPTION_SHARE="$(ros2 pkg prefix bluerov2_description)/share/bluerov2_description"
sed -i "s|package://bluerov2_description|${DESCRIPTION_SHARE}|g" "$URDF_PATH"
echo "[sim_robot] URDF zapisany: $URDF_PATH (package:// → ścieżki absolutne)"

# Isaac Sim
ISAAC_SCRIPT="$(ros2 pkg prefix robot_bringup)/share/robot_bringup/isaac/isaac_sim.py"

if [ "$VERBOSE" -eq 1 ]; then
    python3 "$ISAAC_SCRIPT" &
else
    python3 "$ISAAC_SCRIPT" 2>&1 | grep -v "^\[Warning\]\|^Kit\|^Shader\|^\[omni" &
fi

ISAAC_PID=$!
echo "[sim_robot] Isaac Sim PID: $ISAAC_PID"

sleep 5

ros2 launch robot_bringup isaac.launch.py &
LAUNCH_PID=$!

trap "kill $ISAAC_PID $LAUNCH_PID 2>/dev/null" EXIT
wait $ISAAC_PID
