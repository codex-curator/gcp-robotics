"""
Tests for the batch registry builder pipeline.

Validates each stage independently and the full end-to-end pipeline,
ensuring that changes to the SDK don't silently break registry generation.
"""

import json
import sys
from pathlib import Path

import pytest

# Ensure the SDK and examples are importable
_SDK_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SDK_ROOT))
sys.path.insert(0, str(_SDK_ROOT / "examples"))

from gcp_robotics.hash_engine.hasher import CompositeHasher
from gcp_robotics.hash_engine.registry import CodexRegistry
from gcp_robotics.soulmark import SoulmarkVerifier

from batch_registry_builder import (
    GEOMETRIC_PRIMITIVES,
    generate_primitive_images,
    load_template,
    stage_organize,
    stage_hash,
    stage_enrich,
    stage_sign,
    stage_register,
    stage_validate,
    run_pipeline,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def output_dir(tmp_path):
    return tmp_path / "test_output"


@pytest.fixture
def hasher():
    return CompositeHasher()


@pytest.fixture
def registry():
    return CodexRegistry()


@pytest.fixture
def primitives():
    """Return a small subset of geometric primitives for fast testing."""
    return {
        k: v
        for k, v in list(GEOMETRIC_PRIMITIVES.items())[:2]
    }


# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------


class TestTemplateLoading:
    def test_load_standard_evaluation(self):
        objects = load_template("standard_evaluation")
        assert len(objects) == 5
        assert "cube_red_50mm" in objects
        assert "sphere_blue_60mm" in objects
        assert "l_bracket_white_80x60mm" in objects

    def test_load_ycb_20(self):
        objects = load_template("ycb_20")
        assert len(objects) == 20
        assert "001_chips_can" in objects
        assert "052_extra_large_clamp" in objects

    def test_load_unknown_template_raises(self):
        with pytest.raises(ValueError, match="Unknown template"):
            load_template("nonexistent_template")

    def test_primitives_have_required_fields(self):
        for oid, obj in GEOMETRIC_PRIMITIVES.items():
            assert "name" in obj, f"{oid} missing name"
            assert "mass_kg" in obj, f"{oid} missing mass_kg"
            assert "dimensions_m" in obj, f"{oid} missing dimensions_m"
            assert "material" in obj, f"{oid} missing material"
            assert "shape" in obj, f"{oid} missing shape"
            assert len(obj["dimensions_m"]) == 3, f"{oid} dimensions must be [x,y,z]"


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------


class TestImageGeneration:
    def test_generates_correct_count(self, output_dir):
        images = generate_primitive_images(
            "cube_red_50mm", GEOMETRIC_PRIMITIVES["cube_red_50mm"],
            output_dir, count=3,
        )
        assert len(images) == 3

    def test_images_are_png(self, output_dir):
        images = generate_primitive_images(
            "sphere_blue_60mm", GEOMETRIC_PRIMITIVES["sphere_blue_60mm"],
            output_dir, count=2,
        )
        for img_path in images:
            assert img_path.suffix == ".png"
            assert img_path.exists()
            assert img_path.stat().st_size > 0

    def test_all_shapes_render(self, output_dir):
        """Every shape type should produce valid images."""
        for oid, obj in GEOMETRIC_PRIMITIVES.items():
            images = generate_primitive_images(oid, obj, output_dir, count=1)
            assert len(images) == 1, f"Failed to generate image for {oid}"
            assert images[0].exists()


# ---------------------------------------------------------------------------
# Stage 1: ORGANIZE
# ---------------------------------------------------------------------------


class TestStageOrganize:
    def test_creates_manifest(self, primitives, output_dir):
        manifest = stage_organize(primitives, output_dir)
        assert len(manifest) == 2
        for oid, entry in manifest.items():
            assert entry["status"] == "organized"
            assert entry["object_id"] == oid
            assert "name" in entry

    def test_creates_directories(self, primitives, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = stage_organize(primitives, output_dir)
        for oid in primitives:
            assert (output_dir / oid).is_dir()

    def test_dry_run_no_directories(self, primitives, output_dir):
        manifest = stage_organize(primitives, output_dir, dry_run=True)
        assert len(manifest) == 2
        for oid in primitives:
            assert not (output_dir / oid).exists()


# ---------------------------------------------------------------------------
# Stage 2: HASH
# ---------------------------------------------------------------------------


class TestStageHash:
    def test_produces_hashes(self, primitives, output_dir, hasher):
        manifest = stage_organize(primitives, output_dir)
        manifest = stage_hash(manifest, output_dir, hasher, images_per_object=2)
        for oid, entry in manifest.items():
            assert entry["status"] == "hashed"
            assert "dhash_hex" in entry["hashes"]
            assert "phash_hex" in entry["hashes"]
            assert "content_sha256" in entry["hashes"]
            assert len(entry["images"]) == 2

    def test_hashes_are_hex_strings(self, primitives, output_dir, hasher):
        manifest = stage_organize(primitives, output_dir)
        manifest = stage_hash(manifest, output_dir, hasher, images_per_object=1)
        for entry in manifest.values():
            dhash = entry["hashes"]["dhash_hex"]
            assert dhash.startswith("0x"), f"dHash should start with 0x: {dhash}"
            assert len(dhash) == 18, f"dHash should be 18 chars (0x + 16 hex): {dhash}"

    def test_different_objects_different_hashes(self, output_dir, hasher):
        objects = {
            k: v for k, v in list(GEOMETRIC_PRIMITIVES.items())[:3]
        }
        manifest = stage_organize(objects, output_dir)
        manifest = stage_hash(manifest, output_dir, hasher, images_per_object=1)
        hashes = [e["hashes"]["dhash_hex"] for e in manifest.values()]
        assert len(set(hashes)) == len(hashes), "All objects should have unique hashes"


# ---------------------------------------------------------------------------
# Stage 3: ENRICH
# ---------------------------------------------------------------------------


class TestStageEnrich:
    def test_produces_skbs(self, primitives, output_dir, hasher):
        manifest = stage_organize(primitives, output_dir)
        manifest = stage_hash(manifest, output_dir, hasher, images_per_object=1)
        manifest = stage_enrich(manifest)
        for entry in manifest.values():
            assert entry["status"] == "enriched"
            skb = entry["skb"]
            assert skb["schema_version"] == "2.0-GCP-ROBOTICS"
            assert "layer_1_provenance" in skb
            assert "layer_2_semantic_topology" in skb
            assert "layer_3_affordance" in skb
            assert "layer_4_safety" in skb

    def test_skb_has_correct_classification(self, primitives, output_dir, hasher):
        manifest = stage_organize(primitives, output_dir)
        manifest = stage_hash(manifest, output_dir, hasher, images_per_object=1)
        manifest = stage_enrich(manifest)
        for entry in manifest.values():
            l2 = entry["skb"]["layer_2_semantic_topology"]
            assert l2["object_classification"] is not None
            assert l2["physical_properties"]["estimated_mass_kg"] > 0

    def test_grip_force_in_range(self, output_dir, hasher):
        manifest = stage_organize(GEOMETRIC_PRIMITIVES, output_dir)
        manifest = stage_hash(manifest, output_dir, hasher, images_per_object=1)
        manifest = stage_enrich(manifest)
        for entry in manifest.values():
            force = entry["skb"]["layer_3_affordance"]["action_schema"]["parameters"]["force_profile"]["max_grip_newtons"]
            assert 10.0 <= force <= 50.0, f"Grip force {force}N outside [10, 50] range"


# ---------------------------------------------------------------------------
# Stage 4: SIGN
# ---------------------------------------------------------------------------


class TestStageSign:
    def test_produces_soulmarks(self, primitives, output_dir, hasher):
        manifest = stage_organize(primitives, output_dir)
        manifest = stage_hash(manifest, output_dir, hasher, images_per_object=1)
        manifest = stage_enrich(manifest)
        manifest = stage_sign(manifest)
        for entry in manifest.values():
            assert entry["status"] == "signed"
            assert len(entry["soulmark"]) == 64  # SHA-256 hex

    def test_soulmark_written_to_skb(self, primitives, output_dir, hasher):
        manifest = stage_organize(primitives, output_dir)
        manifest = stage_hash(manifest, output_dir, hasher, images_per_object=1)
        manifest = stage_enrich(manifest)
        manifest = stage_sign(manifest)
        for entry in manifest.values():
            skb_soulmark = entry["skb"]["triple_hash"]["soulmark_sha256"]
            assert skb_soulmark == entry["soulmark"]
            assert skb_soulmark != "pending"

    def test_hmac_with_fleet_secret(self, primitives, output_dir, hasher):
        manifest = stage_organize(primitives, output_dir)
        manifest = stage_hash(manifest, output_dir, hasher, images_per_object=1)
        manifest = stage_enrich(manifest)
        manifest = stage_sign(manifest, fleet_secret="test-fleet-key")
        for entry in manifest.values():
            assert entry["soulmark_method"] == "HMAC-SHA256"

    def test_different_secrets_different_soulmarks(self, primitives, output_dir, hasher):
        # Run with two different secrets
        m1 = stage_organize(primitives, output_dir)
        m1 = stage_hash(m1, output_dir, hasher, images_per_object=1)
        m1 = stage_enrich(m1)
        m1 = stage_sign(m1, fleet_secret="key-a")

        m2 = stage_organize(primitives, output_dir)
        m2 = stage_hash(m2, output_dir, hasher, images_per_object=1)
        m2 = stage_enrich(m2)
        m2 = stage_sign(m2, fleet_secret="key-b")

        for oid in primitives:
            assert m1[oid]["soulmark"] != m2[oid]["soulmark"]


# ---------------------------------------------------------------------------
# Stage 5: REGISTER
# ---------------------------------------------------------------------------


class TestStageRegister:
    def test_populates_registry(self, primitives, output_dir, hasher, registry):
        manifest = stage_organize(primitives, output_dir)
        manifest = stage_hash(manifest, output_dir, hasher, images_per_object=2)
        manifest = stage_enrich(manifest)
        manifest = stage_sign(manifest)
        manifest = stage_register(manifest, registry, hasher)
        for entry in manifest.values():
            assert entry["status"] == "registered"
            assert entry["lookup_verified"] is True
            assert entry["lookup_time_ms"] < 1.0

    def test_registry_entry_count(self, primitives, output_dir, hasher, registry):
        manifest = stage_organize(primitives, output_dir)
        manifest = stage_hash(manifest, output_dir, hasher, images_per_object=3)
        manifest = stage_enrich(manifest)
        manifest = stage_sign(manifest)
        manifest = stage_register(manifest, registry, hasher)
        stats = registry.stats()
        # Each object has 1 canonical + 2 additional = 3 entries per object
        assert stats["total_entries"] >= len(primitives)

    def test_round_trip_lookup(self, output_dir, hasher, registry):
        objects = {"cube_red_50mm": GEOMETRIC_PRIMITIVES["cube_red_50mm"]}
        manifest = stage_organize(objects, output_dir)
        manifest = stage_hash(manifest, output_dir, hasher, images_per_object=1)
        manifest = stage_enrich(manifest)
        manifest = stage_sign(manifest)
        manifest = stage_register(manifest, registry, hasher)

        dhash = manifest["cube_red_50mm"]["hashes"]["dhash_hex"]
        result = registry.lookup(dhash)
        assert result.match_type == "EXACT"
        assert result.skb_data["schema_version"] == "2.0-GCP-ROBOTICS"


# ---------------------------------------------------------------------------
# Stage 6: VALIDATE
# ---------------------------------------------------------------------------


class TestStageValidate:
    def test_generates_protocol(self, primitives, output_dir, hasher, registry):
        manifest = stage_organize(primitives, output_dir)
        manifest = stage_hash(manifest, output_dir, hasher, images_per_object=1)
        manifest = stage_enrich(manifest)
        manifest = stage_sign(manifest)
        manifest = stage_register(manifest, registry, hasher)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = stage_validate(manifest, output_dir)

        protocol_path = output_dir / "experiment_protocol.json"
        assert protocol_path.exists()
        protocol = json.loads(protocol_path.read_text())
        assert protocol["summary"]["total_phases"] == 4
        assert protocol["summary"]["total_trials"] > 0


# ---------------------------------------------------------------------------
# Full pipeline end-to-end
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_standard_evaluation_end_to_end(self, tmp_path):
        out = str(tmp_path / "std_eval")
        report = run_pipeline(
            template_name="standard_evaluation",
            output_dir=out,
            images_per_object=1,
        )
        assert report["batch_report"]["success_rate"] == 100.0
        assert report["batch_report"]["total_objects"] == 5

        # Verify output files
        out_path = Path(out)
        assert (out_path / "standard_evaluation_registry.json").exists()
        assert (out_path / "experiment_protocol.json").exists()
        assert (out_path / "skbs").is_dir()
        skbs = list((out_path / "skbs").glob("*.json"))
        assert len(skbs) == 5

    def test_registry_is_loadable(self, tmp_path):
        out = str(tmp_path / "load_test")
        run_pipeline(
            template_name="standard_evaluation",
            output_dir=out,
            images_per_object=1,
        )
        # Load the saved registry and verify lookups still work
        registry = CodexRegistry()
        registry.load(str(Path(out) / "standard_evaluation_registry.json"))
        stats = registry.stats()
        assert stats["total_entries"] >= 5

    def test_skb_json_is_valid(self, tmp_path):
        out = str(tmp_path / "json_test")
        run_pipeline(
            template_name="standard_evaluation",
            output_dir=out,
            images_per_object=1,
        )
        skbs_dir = Path(out) / "skbs"
        for skb_file in skbs_dir.glob("*.json"):
            data = json.loads(skb_file.read_text())
            assert data["schema_version"] == "2.0-GCP-ROBOTICS"
            assert "layer_1_provenance" in data
            assert "layer_2_semantic_topology" in data
            assert "layer_3_affordance" in data
            assert "layer_4_safety" in data
            assert data["triple_hash"]["soulmark_sha256"] != "pending"

    def test_dry_run_produces_no_files(self, tmp_path):
        out = str(tmp_path / "dry_run")
        report = run_pipeline(
            template_name="standard_evaluation",
            output_dir=out,
            dry_run=True,
        )
        assert not (Path(out) / "standard_evaluation_registry.json").exists()
        assert not (Path(out) / "skbs").exists()
