#!/usr/bin/env python3
"""Utility script to sanity-check whether a USD spawns as a single articulation."""

from __future__ import annotations

import argparse
import copy
import importlib
import sys

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--usd-path",
        required=True,
        help="Absolute path to the USD that should contain the mobile manipulator.",
    )
    parser.add_argument(
        "--root-prim-path",
        required=True,
        help="Prim path (inside the USD) where UsdPhysics.ArticulationRootAPI is applied.",
    )
    parser.add_argument(
        "--prim-path",
        default="/World/Robot",
        help="Scene prim where the USD will be spawned (default: /World/Robot).",
    )
    parser.add_argument(
        "--activate-contact-sensors",
        action="store_true",
        help="Enable contact sensor API on all rigid bodies while spawning.",
    )
    parser.add_argument(
        "--cfg-entry",
        help="Optional config entry point in the form 'module.path:ATTRIBUTE'. If provided, the articulation cfg "
        "is cloned from this entry (recommended).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run SimulationApp in headless mode.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app

    import isaaclab.sim as sim_utils  # noqa: WPS433
    from isaaclab.assets import ArticulationCfg  # noqa: WPS433
    from isaaclab.sim import SimulationContext  # noqa: WPS433

    if args.cfg_entry is not None:
        try:
            module_path, attr_name = args.cfg_entry.split(":")
        except ValueError as exc:
            raise ValueError("--cfg-entry must be in the form 'module.path:ATTRIBUTE'") from exc
        cfg_module = importlib.import_module(module_path)
        cfg_template = getattr(cfg_module, attr_name)
        cfg = copy.deepcopy(cfg_template)
        cfg.spawn.usd_path = args.usd_path
        cfg.prim_path = args.prim_path
        cfg.articulation_root_prim_path = args.root_prim_path
        if args.activate_contact_sensors:
            cfg.spawn.activate_contact_sensors = True
    else:
        cfg = ArticulationCfg(
            prim_path=args.prim_path,
            articulation_root_prim_path=args.root_prim_path,
            spawn=sim_utils.UsdFileCfg(
                usd_path=args.usd_path,
                activate_contact_sensors=args.activate_contact_sensors,
            ),
        )

    sim_context = SimulationContext()
    sim_context.clear()

    print("Spawning USD:", args.usd_path)
    print("Spawning prim path:", args.prim_path)
    print("Articulation root prim path:", args.root_prim_path)

    try:
        asset = cfg.class_type(cfg)
        sim_context.reset()
        if not asset.is_initialized:
            raise RuntimeError("Asset failed to initialize (is_initialized=False).")
        # Touch PhysX views to ensure tensors are valid.
        _ = asset.num_bodies
        _ = asset.num_joints
    except Exception as exc:
        print("✘ Failed to initialize articulation:", exc)
        simulation_app.close()
        return 1

    print("✔ Articulation initialized")
    print("  - num bodies :", asset.num_bodies)
    print("  - num joints :", asset.num_joints)

    simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
