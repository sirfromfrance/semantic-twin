#!/bin/bash
# Low inspection route (8 m, the launch default) in the changed world: the route used for the people flights.
# usage: bash tools/fly_inspection.sh [gui:=false] [world_file]
source /opt/ros/humble/setup.bash
source ~/ROS/thesis_ws/install/setup.bash
GUI=${1:-gui:=true}
WORLD=${2:-disaster_scene_changed.world}
exec ros2 launch semantic_twin disaster_launch.xml world_file:=$WORLD $GUI
