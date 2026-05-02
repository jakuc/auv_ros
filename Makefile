COMPOSE = docker compose -f docker/docker-compose.yml


enter:
	xhost +local:docker > /dev/null
	@if ! docker ps -a -q -f name=auv_isaac | grep -q .; then \
		$(COMPOSE) up -d; \
	elif ! docker ps -q -f name=auv_isaac | grep -q .; then \
		docker start auv_isaac; \
	fi
	docker exec -it -e OMNI_KIT_ALLOW_ROOT=1 -w /workspace auv_isaac bash

build:
	$(COMPOSE) build

colcon:
	@if ! docker ps -a -q -f name=auv_isaac | grep -q .; then \
		$(COMPOSE) up -d; \
	elif ! docker ps -q -f name=auv_isaac | grep -q .; then \
		docker start auv_isaac; \
	fi
	docker exec -it auv_isaac bash -c "source /opt/ros/humble/setup.bash && source /opt/ros_ws/install/setup.bash 2>/dev/null || true && cd /workspace && colcon build --symlink-install"

.PHONY: enter build colcon
