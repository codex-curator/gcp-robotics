"""Shared test fixtures for gcp-robotics test suite."""

import json
import pytest
from gcp_robotics.cache import FastPathCache
from gcp_robotics.hash_engine.hasher import CompositeHash
from gcp_robotics.hash_engine.registry import CodexRegistry
from gcp_robotics.soulmark import SoulmarkVerifier
from gcp_robotics.telemetry import TelemetryLogger


@pytest.fixture
def sample_skb():
    """A minimal valid SKB payload for testing."""
    return {
        "provenance": {
            "skb_id": "SKB-TEST-001",
            "object_canonical_name": "red_bolt_m6",
            "workspace_id": "assembly_cell_A",
            "soulmark": None,
        },
        "semantic_topology": {
            "perceptual_hashes": {
                "dhash_64": "0xA1B2C3D4E5F60718",
                "phash_64": "0x1234567890ABCDEF",
            },
            "physical_properties": {
                "mass_kg": 0.015,
                "material": "steel",
                "dimensions_mm": {"length": 30, "width": 10, "height": 10},
            },
        },
        "affordance_layer": {
            "action_primitive": "PICK_AND_PLACE",
            "grasp_type": "parallel_jaw",
            "grip_force_n": 5.0,
            "approach_vector": [0, 0, -1],
            "target_pose_se3": {
                "position": [0.3, 0.1, 0.05],
                "orientation_quat": [1, 0, 0, 0],
            },
        },
        "safety_layer": {
            "hazard_class": "NONE",
            "max_force_n": 20.0,
            "max_velocity_ms": 0.5,
            "cbf_radius_m": 0.05,
        },
    }


@pytest.fixture
def sample_composite_hash():
    """A sample CompositeHash for testing."""
    return CompositeHash(
        dhash_hex="0xA1B2C3D4E5F60718",
        phash_hex="0x1234567890ABCDEF",
        ppp_256bit="0x" + "AB" * 32,
        color_hash="0x" + "FF8040" * 16,
    )


@pytest.fixture
def cache():
    """A fresh FastPathCache with small size for testing."""
    return FastPathCache(max_entries=100)


@pytest.fixture
def registry():
    """A fresh CodexRegistry."""
    return CodexRegistry()


@pytest.fixture
def verifier():
    """A SoulmarkVerifier with no fleet secret."""
    return SoulmarkVerifier()


@pytest.fixture
def verifier_with_secret():
    """A SoulmarkVerifier with a fleet secret."""
    return SoulmarkVerifier(fleet_secret="test-fleet-secret-key")


@pytest.fixture
def telemetry():
    """A TelemetryLogger for testing."""
    return TelemetryLogger(enabled=True, buffer_size=100)
