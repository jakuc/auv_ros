#!/bin/bash
set -e

# Warstwa 1: ROS2 Humble (baza)
source /opt/ros/humble/setup.bash

# Warstwa 2: pre-built workspace z obrazu (fallback dla nowego sprzętu)
if [ -f /opt/ros_ws/install/setup.bash ]; then
    source /opt/ros_ws/install/setup.bash
fi

# Warstwa 3: workspace developerski z volume (nadpisuje layer 2 jeśli istnieje)
if [ -f /workspace/install/setup.bash ]; then
    source /workspace/install/setup.bash
fi

# Poprawka: ament_cmake pakiety nie trafiają automatycznie do AMENT_PREFIX_PATH.
# Skanuj oba workspace'y i dodaj brakujące pakiety.
for ws_install in /opt/ros_ws/install /workspace/install; do
    if [ -d "$ws_install" ]; then
        for pkg_dir in "$ws_install"/*/; do
            if [ -d "${pkg_dir}share/ament_index" ]; then
                pkg_path="${pkg_dir%/}"
                case ":$AMENT_PREFIX_PATH:" in
                    *":$pkg_path:"*) ;;
                    *) export AMENT_PREFIX_PATH="$pkg_path:$AMENT_PREFIX_PATH" ;;
                esac
            fi
        done
    fi
done

# X11 GUI - pozwól rootowi na połączenie jeśli DISPLAY ustawiony
if [ -n "$DISPLAY" ]; then
    xhost +local:root 2>/dev/null || true
fi

exec "$@"
