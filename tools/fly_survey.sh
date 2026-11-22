#!/bin/bash
# Starts the simulation, PX4, MAVROS and the mission (survey route, 12 m, 0.7 m/s) in the changed world.
# usage: bash tools/fly_survey.sh [gui:=false] [world_file]     (default: gui on, disaster_scene_changed.world)
source /opt/ros/humble/setup.bash
source ~/ROS/thesis_ws/install/setup.bash
GUI=${1:-gui:=true}
WORLD=${2:-disaster_scene_changed.world}
exec ros2 launch semantic_twin disaster_launch.xml world_file:=$WORLD $GUI \
  waypoints:="[-10.0, -4.0, 12.0, -19.0, -6.0, 12.0, 4.5, -6.0, 12.0, 5.0, -5.0, 12.0, 8.0, -3.0, 12.0, 12.0, -4.0, 12.0, 13.5, -6.0, 12.0, 20.0, -6.0, 12.0, 20.0, 3.0, 12.0, -4.0, 3.0, 12.0, 0.0, 9.0, 12.0, 0.0, 12.0, 12.0, 20.0, 12.0, 12.0, 0.0, 0.0, 12.0]" \
  takeoff_alt:=12.0 max_speed:=0.7
