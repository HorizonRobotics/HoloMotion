#!/usr/bin/env python3
##############################################################################
# close_ai_sport.py
#
# Purpose:
#   Pre-launch helper to switch off Unitree "ai_sport" so the robot enters
#   debug mode automatically at startup. Idempotent; failure will not block
#   the launch (logs and exits 0).
#
# How it works:
#   1) Initialize Unitree SDK2 channel on given NIC (default: eth0)
#   2) Call RobotStateClient.ServiceSwitch("ai_sport", False)
#   3) Print the Python interpreter path for environment diagnostics
#
# Prerequisites:
#   - unitree_sdk2py (Python SDK2)
#     Example (editable install):
#       python3 -m pip install --user -e /path/to/unitree_sdk2_python
#     Or install vendor-provided wheel/package.
#
# Usage:
#   python3 close_ai_sport.py eth0
#   # If interface is omitted, it defaults to "eth0".
#
# Notes:
#   - Safe to call repeatedly
#   - Falls back to go2/b2 clients if g1 client is unavailable
#   - If unitree_sdk2py import fails, ensure the interpreter used by launch
#     can find it (site-packages or PYTHONPATH).
#
# Author: HoloMotion Team
# License: See project LICENSE file
##############################################################################

import sys
import time
print("[close_ai_sport] sys.executable:", sys.executable)
def main():
    iface = sys.argv[1] if len(sys.argv) > 1 else "eth0"
    try:
        # Prefer G1 client
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        try:
            from unitree_sdk2py.g1.robot_state.robot_state_client import RobotStateClient
        except ImportError:
            # Fallbacks (some installs place go2/b2 clients only)
            try:
                from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient
            except ImportError:
                from unitree_sdk2py.b2.robot_state.robot_state_client import RobotStateClient

        ChannelFactoryInitialize(0, iface)
        rsc = RobotStateClient()
        rsc.SetTimeout(3.0)
        rsc.Init()
        # Switch off ai_sport
        code = rsc.ServiceSwitch("ai_sport", False)
        if code == 0:
            print("[close_ai_sport] ai_sport off: success")
            return 0
        else:
            print(f"[close_ai_sport] ai_sport off: failed code={code}")
            return 0  # do not block launch
    except Exception as e:
        print(f"[close_ai_sport] exception: {e}")
        return 0  # do not block launch

if __name__ == "__main__":
    sys.exit(main())


