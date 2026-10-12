#!/bin/bash
# Start one PX4 SITL instance. Usage: px4.sh <instance>
# PX4 location can be overridden with PX4_DIR.
PX4_DIR="${PX4_DIR:-$HOME/ROS/PX4-Autopilot}"
BUILD="$PX4_DIR/build/px4_sitl_default"
SIM_DIR="$HOME/.ros/sitl_iris_$1"
mkdir -p "$SIM_DIR"

cat << RC > "$SIM_DIR/custom_rc.sh"
#!/bin/sh
. etc/init.d-posix/rcS

param set COM_RC_IN_MODE 1
param set COM_RCL_EXCEPT 4
param set NAV_RCL_ACT 0
param set NAV_DLL_ACT 0
param set SIM_BAT_DRAIN 0
param set COM_ARM_BAT_MIN 0

RC

chmod +x "$SIM_DIR/custom_rc.sh"
cd ~/.ros
"$BUILD/bin/px4" "$BUILD/etc" -s "$SIM_DIR/custom_rc.sh" -i "$1" -w "$SIM_DIR"
exit 0
