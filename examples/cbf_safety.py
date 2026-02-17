#!/usr/bin/env python3
"""CBF-QP Safety Constraint Demo for GCP-Robotics SDK.

This demonstrates the three-layer safety defense described in the PCO
"mechanical fuse" architecture:

    (1) SKB force/speed limits  -- declarative constraints encoded in the
        SafetyLayer and ForceProfile of a Spatial Kinematic Blueprint.
    (2) CBF-QP real-time constraint enforcement  -- Control Barrier Functions
        project unsafe commanded actions onto the boundary of the safe set,
        clamping force, speed, and workspace violations in real time.
    (3) Soulmark integrity verification  -- SHA-256 / HMAC-SHA256 digest
        ensures the SKB has not been tampered with before any constraints
        are trusted.

Run:
    python examples/cbf_safety.py

No hardware, API keys, or solver libraries required (uses numpy only).
"""

from __future__ import annotations

import numpy as np

from gcp_robotics.schema.models import (
    ForceProfile,
    HazardClass,
    OperatingConstraints,
    SafetyLayer,
)
from gcp_robotics.soulmark import SoulmarkVerifier


# ── 1. Define SKB safety constraints ────────────────────────────────────

force_profile = ForceProfile(
    max_grip_newtons=25.0,
    lateral_torque_limit_nm=1.2,
)

operating_constraints = OperatingConstraints(
    max_speed_mm_s=200.0,
    max_force_n=30.0,
    restricted_zones=["zone_A3"],
)

safety_layer = SafetyLayer(
    hazard_class=HazardClass.PINCH_POINT,
    ppe_required=["safety_glasses"],
    regulatory_certifications=["ISO 10218-1"],
    operating_constraints=operating_constraints,
)

# Workspace boundary: axis-aligned box [x_min, x_max, y_min, y_max] in mm
WORKSPACE_BOUNDS = np.array([0.0, 500.0, 0.0, 400.0])


# ── 2. CBF-QP safety filter ────────────────────────────────────────────

def cbf_barrier_values(
    position: np.ndarray,
    force_cmd: np.ndarray,
    speed_cmd: float,
) -> dict[str, float]:
    """Evaluate Control Barrier Functions h(x) >= 0 for all constraints.

    Returns a dict of barrier name -> barrier value.  Negative values
    indicate a constraint violation that the QP must correct.
    """
    grip_force = np.linalg.norm(force_cmd[:3])
    lateral_torque = abs(force_cmd[3]) if len(force_cmd) > 3 else 0.0

    return {
        "h_grip_force": force_profile.max_grip_newtons - grip_force,
        "h_lateral_torque": force_profile.lateral_torque_limit_nm - lateral_torque,
        "h_speed": operating_constraints.max_speed_mm_s - speed_cmd,
        "h_ws_x_lo": position[0] - WORKSPACE_BOUNDS[0],
        "h_ws_x_hi": WORKSPACE_BOUNDS[1] - position[0],
        "h_ws_y_lo": position[1] - WORKSPACE_BOUNDS[2],
        "h_ws_y_hi": WORKSPACE_BOUNDS[3] - position[1],
    }


def cbf_qp_filter(
    position: np.ndarray,
    force_cmd: np.ndarray,
    speed_cmd: float,
) -> tuple[np.ndarray, float]:
    """Project commands onto the safe set via simple barrier clamping.

    This is a projection-based approximation of the full CBF-QP: each
    violated barrier is enforced independently by clamping the
    corresponding command component to its limit.

    Returns the (safe_force, safe_speed) after projection.
    """
    safe_force = force_cmd.copy()
    safe_speed = speed_cmd

    # Force magnitude clamp
    grip = np.linalg.norm(safe_force[:3])
    if grip > force_profile.max_grip_newtons:
        safe_force[:3] *= force_profile.max_grip_newtons / grip

    # Lateral torque clamp
    if len(safe_force) > 3 and abs(safe_force[3]) > force_profile.lateral_torque_limit_nm:
        safe_force[3] = np.sign(safe_force[3]) * force_profile.lateral_torque_limit_nm

    # Speed clamp
    if safe_speed > operating_constraints.max_speed_mm_s:
        safe_speed = operating_constraints.max_speed_mm_s

    # Workspace position clamp (nudge toward interior)
    position[0] = np.clip(position[0], WORKSPACE_BOUNDS[0], WORKSPACE_BOUNDS[1])
    position[1] = np.clip(position[1], WORKSPACE_BOUNDS[2], WORKSPACE_BOUNDS[3])

    return safe_force, safe_speed


# ── 3. Soulmark verification ───────────────────────────────────────────

def verify_skb_integrity(skb_data: dict, soulmark: str) -> bool:
    """Verify the SKB soulmark before trusting its constraints."""
    verifier = SoulmarkVerifier(fleet_secret="demo-fleet-key")
    result = verifier.verify(skb_data, soulmark)
    return result.valid


# ── 4. Run simulation ──────────────────────────────────────────────────

def main() -> None:
    print("=" * 64)
    print("  CBF-QP Safety Constraint Demo  --  GCP-Robotics SDK")
    print("=" * 64)

    # Layer 3: Soulmark verification
    skb_payload = {
        "safety_layer": safety_layer.model_dump(mode="json"),
        "affordance_layer": {"force_profile": force_profile.model_dump(mode="json")},
    }
    verifier = SoulmarkVerifier(fleet_secret="demo-fleet-key")
    signed = verifier.sign(skb_payload)
    result = verifier.verify(skb_payload, signed.soulmark)
    print(f"\n[Soulmark]  integrity={result.valid}  hash={signed.soulmark[:24]}...")
    if not result.valid:
        print("  ABORT: SKB integrity check failed. Cannot trust constraints.")
        return

    # Scenarios: each is (position, force_cmd, speed_cmd, label)
    scenarios = [
        (np.array([250.0, 200.0]), np.array([10.0, 5.0, 3.0, 0.5]), 150.0,
         "Normal operation (within limits)"),
        (np.array([250.0, 200.0]), np.array([20.0, 15.0, 10.0, 2.5]), 350.0,
         "Excessive force + speed"),
        (np.array([550.0, -20.0]), np.array([5.0, 3.0, 2.0, 0.3]), 100.0,
         "Outside workspace boundary"),
    ]

    for pos, force, speed, label in scenarios:
        print(f"\n--- {label} ---")
        barriers = cbf_barrier_values(pos, force, speed)
        violations = {k: v for k, v in barriers.items() if v < 0}

        print(f"  Commanded: force={force}, speed={speed:.0f} mm/s, pos={pos}")

        if violations:
            print(f"  Barriers violated: {violations}")
        else:
            print("  All barriers h(x) >= 0  [SAFE]")

        safe_force, safe_speed = cbf_qp_filter(pos, force, speed)
        print(f"  After CBF: force={np.round(safe_force, 2)}, "
              f"speed={safe_speed:.0f} mm/s, pos={pos}")

    print("\n" + "=" * 64)
    print("  Three-layer defense verified: SKB limits | CBF-QP | Soulmark")
    print("=" * 64)


if __name__ == "__main__":
    main()
