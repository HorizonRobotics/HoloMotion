"""Hand off the official motion controller before low-level control starts."""

import argparse, json, os, sys, time
from pathlib import Path


def acquire_low_level(client, attempts=3, pause=time.sleep):
    for attempt in range(attempts + 1):
        code, mode = client.CheckMode()
        if code != 0 or not isinstance(mode, dict) or "name" not in mode:
            raise RuntimeError(
                f"Cannot confirm official motion mode: code={code}, mode={mode!r}"
            )
        print(
            "[Control] official motion mode: " + json.dumps(mode), flush=True
        )
        if mode["name"] == "":
            print(
                "[Control] official motion control released; low-level controller may start.",
                flush=True,
            )
            return
        if attempt == attempts:
            raise RuntimeError(
                "Official motion mode is still active; refusing to start low-level control."
            )
        code, _ = client.ReleaseMode()
        if code != 0:
            raise RuntimeError(
                f"Official motion mode release failed: code={code}"
            )
        pause(0.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--isolated-test", action="store_true")
    parser.add_argument("--interface")
    parser.add_argument(
        "--sdk-root",
        default=os.environ.get(
            "UNITREE_SDK2_PYTHON_ROOT", "/root/unitree_sdk2_python"
        ),
    )
    deploy = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--profile", default=str(deploy / "launch_profiles/orin_docker.yaml")
    )
    parser.add_argument(
        "--profile-override", "--set", action="append", default=[]
    )
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    if args.isolated_test:
        interfaces = {p.name for p in Path("/sys/class/net").iterdir()}
        if interfaces != {"lo"}:
            raise RuntimeError(
                f"Isolated test requires only loopback network, found {interfaces}"
            )
        print(
            "[Control] isolated test network verified; no robot service calls.",
            flush=True,
        )
        return
    sys.path.insert(0, str(deploy / "src"))
    from humanoid_policy.launch_profile import load_effective_profile

    _, profile = load_effective_profile(
        args.profile,
        override_arg=";".join(args.profile_override),
        roots=[deploy, Path.cwd()],
    )
    interface = args.interface or profile["robot"]["network_interface"]
    sys.path.insert(0, args.sdk_root)
    from unitree_sdk2py.core import channel
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
        MotionSwitcherClient,
    )

    channel.ChannelConfigHasInterface = '<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="$__IF_NAME__$"/></Interfaces></General></Domain></CycloneDDS>'
    channel.ChannelFactoryInitialize(0, interface)
    client = MotionSwitcherClient()
    client.SetTimeout(3.0)
    client.Init()
    if args.check_only:
        code, mode = client.CheckMode()
        if code != 0 or not isinstance(mode, dict) or "name" not in mode:
            raise RuntimeError(
                f"Cannot confirm official motion mode: code={code}, mode={mode!r}"
            )
        print(
            "[Control] read-only check: "
            + json.dumps(
                {"code": code, "mode": mode, "release_mode_called": False}
            ),
            flush=True,
        )
        return
    acquire_low_level(client)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            "[Control] startup blocked: " + str(exc),
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)
