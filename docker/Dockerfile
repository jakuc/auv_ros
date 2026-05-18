FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Warsaw
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# =============================================================================
# Warstwa 1: Zależności systemowe
# =============================================================================
RUN apt-get update && apt-get install -y \
    curl wget git gnupg2 lsb-release \
    software-properties-common \
    python3 python3-pip python3-venv \
    build-essential cmake \
    nano \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Warstwa 2: ROS2 Humble (Ubuntu 22.04 Jammy)
# Uwaga: host używa Jazzy (Ubuntu 24.04), kontener używa Humble (Ubuntu 22.04)
# =============================================================================
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
        http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
        > /etc/apt/sources.list.d/ros2.list && \
    apt-get update && apt-get install -y \
    ros-humble-desktop \
    ros-humble-tf2-ros \
    ros-humble-tf2-geometry-msgs \
    ros-humble-std-srvs \
    ros-humble-sensor-msgs \
    ros-humble-geometry-msgs \
    ros-humble-nav-msgs \
    ros-humble-xacro \
    ros-humble-plotjuggler-ros \
    ros-humble-robot-localization \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-argcomplete \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Warstwa 3: Isaac Sim 4.5.0
# UWAGA: To jest ~25GB pobierania. Pierwsza budowa trwa 30-90 minut.
# Warstwa jest cache'owana - kolejne budowy (po zmianach w src/) są szybkie.
# =============================================================================
RUN pip3 install "isaacsim[all]==4.5.0.0" --extra-index-url https://pypi.nvidia.com

# =============================================================================
# Warstwa 4: Pre-built workspace ROS2
# Budowany w /opt/ros_ws - dostępny jako fallback na nowym sprzęcie
# bez potrzeby robienia colcon build przez użytkownika
# =============================================================================
WORKDIR /opt/ros_ws
COPY src/ src/

RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
    colcon build --symlink-install 2>&1 || true"

# =============================================================================
# Entrypoint
# =============================================================================
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "[ -f /opt/ros_ws/install/setup.bash ] && source /opt/ros_ws/install/setup.bash" >> ~/.bashrc && \
    echo "[ -f /workspace/install/setup.bash ] && source /workspace/install/setup.bash" >> ~/.bashrc

WORKDIR /workspace

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
