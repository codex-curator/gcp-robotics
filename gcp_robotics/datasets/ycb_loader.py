"""
YCBLoader — Ground-truth physical property loader for the YCB Object and Model Set.

The Yale-CMU-Berkeley (YCB) Object and Model Set contains 77 everyday objects
with precisely measured physical properties (mass, dimensions, material).  This
module provides a hardcoded dictionary of the 20 most manipulation-relevant
objects and utilities to:

  - Download or generate test images for each object.
  - Retrieve ground-truth physical properties.
  - Produce SKB (Semantic Knowledge Base) seed dictionaries pre-populated with
    REAL measured data, so the downstream schema pipeline starts from physical
    truth rather than VLM-hallucinated estimates.

Dataset reference: http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/
"""

from __future__ import annotations

import io
import logging
import tarfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests
from PIL import Image, ImageDraw, ImageFont
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# YCB ground-truth object catalogue
# ---------------------------------------------------------------------------

YCB_OBJECTS: Dict[str, dict] = {
    "001_chips_can": {
        "name": "Pringles Chips Can",
        "mass_kg": 0.205,
        "dimensions_m": [0.076, 0.076, 0.25],
        "material": "cardboard_metal",
        "category": "food_container",
        "graspable": True,
    },
    "002_master_chef_can": {
        "name": "Master Chef Coffee Can",
        "mass_kg": 0.414,
        "dimensions_m": [0.102, 0.102, 0.14],
        "material": "metal",
        "category": "food_container",
        "graspable": True,
    },
    "003_cracker_box": {
        "name": "Cheez-It Cracker Box",
        "mass_kg": 0.411,
        "dimensions_m": [0.06, 0.158, 0.21],
        "material": "cardboard",
        "category": "food_box",
        "graspable": True,
    },
    "004_sugar_box": {
        "name": "Domino Sugar Box",
        "mass_kg": 0.514,
        "dimensions_m": [0.038, 0.089, 0.175],
        "material": "cardboard",
        "category": "food_box",
        "graspable": True,
    },
    "005_tomato_soup_can": {
        "name": "Campbell's Tomato Soup Can",
        "mass_kg": 0.349,
        "dimensions_m": [0.068, 0.068, 0.102],
        "material": "metal",
        "category": "food_container",
        "graspable": True,
    },
    "006_mustard_bottle": {
        "name": "French's Mustard Bottle",
        "mass_kg": 0.603,
        "dimensions_m": [0.053, 0.093, 0.19],
        "material": "plastic",
        "category": "condiment",
        "graspable": True,
    },
    "007_tuna_fish_can": {
        "name": "StarKist Tuna Can",
        "mass_kg": 0.171,
        "dimensions_m": [0.085, 0.085, 0.033],
        "material": "metal",
        "category": "food_container",
        "graspable": True,
    },
    "008_pudding_box": {
        "name": "Jell-O Pudding Box",
        "mass_kg": 0.187,
        "dimensions_m": [0.033, 0.089, 0.127],
        "material": "cardboard",
        "category": "food_box",
        "graspable": True,
    },
    "009_gelatin_box": {
        "name": "Jell-O Gelatin Box",
        "mass_kg": 0.097,
        "dimensions_m": [0.028, 0.073, 0.102],
        "material": "cardboard",
        "category": "food_box",
        "graspable": True,
    },
    "010_potted_meat_can": {
        "name": "Hormel Potted Meat Can",
        "mass_kg": 0.370,
        "dimensions_m": [0.058, 0.101, 0.083],
        "material": "metal",
        "category": "food_container",
        "graspable": True,
    },
    "011_banana": {
        "name": "Banana (plastic)",
        "mass_kg": 0.066,
        "dimensions_m": [0.038, 0.18, 0.04],
        "material": "plastic",
        "category": "fruit_model",
        "graspable": True,
    },
    "019_pitcher_base": {
        "name": "Pitcher Base",
        "mass_kg": 0.178,
        "dimensions_m": [0.104, 0.104, 0.098],
        "material": "plastic",
        "category": "kitchen_tool",
        "graspable": True,
    },
    "021_bleach_cleanser": {
        "name": "Soft Scrub Cleanser",
        "mass_kg": 0.302,
        "dimensions_m": [0.05, 0.102, 0.25],
        "material": "plastic",
        "category": "cleaning",
        "graspable": True,
    },
    "024_bowl": {
        "name": "Bowl",
        "mass_kg": 0.147,
        "dimensions_m": [0.16, 0.16, 0.05],
        "material": "plastic",
        "category": "tableware",
        "graspable": True,
    },
    "025_mug": {
        "name": "Mug",
        "mass_kg": 0.118,
        "dimensions_m": [0.117, 0.093, 0.081],
        "material": "ceramic",
        "category": "tableware",
        "graspable": True,
    },
    "035_power_drill": {
        "name": "Power Drill",
        "mass_kg": 0.895,
        "dimensions_m": [0.184, 0.187, 0.052],
        "material": "plastic_metal",
        "category": "tool",
        "graspable": True,
    },
    "036_wood_block": {
        "name": "Wood Block Set",
        "mass_kg": 0.729,
        "dimensions_m": [0.085, 0.085, 0.2],
        "material": "wood",
        "category": "block",
        "graspable": True,
    },
    "037_scissors": {
        "name": "Scissors",
        "mass_kg": 0.082,
        "dimensions_m": [0.027, 0.095, 0.2],
        "material": "metal_plastic",
        "category": "tool",
        "graspable": True,
    },
    "040_large_marker": {
        "name": "Large Marker",
        "mass_kg": 0.016,
        "dimensions_m": [0.024, 0.024, 0.121],
        "material": "plastic",
        "category": "stationery",
        "graspable": True,
    },
    "052_extra_large_clamp": {
        "name": "Extra Large Clamp",
        "mass_kg": 0.202,
        "dimensions_m": [0.052, 0.155, 0.206],
        "material": "metal_plastic",
        "category": "tool",
        "graspable": True,
    },
}


# ---------------------------------------------------------------------------
# Material-based default friction coefficients
# ---------------------------------------------------------------------------

_FRICTION_BY_MATERIAL: Dict[str, float] = {
    "plastic": 0.4,
    "metal": 0.3,
    "cardboard": 0.5,
    "ceramic": 0.35,
    "wood": 0.45,
    # Composite materials — use the average of their constituents.
    "cardboard_metal": 0.4,
    "plastic_metal": 0.35,
    "metal_plastic": 0.35,
}


# ---------------------------------------------------------------------------
# Category-based default action primitives
# ---------------------------------------------------------------------------

_ACTION_PRIMITIVES_BY_CATEGORY: Dict[str, List[str]] = {
    "food_container": ["PICK_VERTICAL", "PLACE_STABLE"],
    "food_box": ["PICK_VERTICAL", "PLACE_STABLE"],
    "condiment": ["PICK_VERTICAL", "PLACE_STABLE", "POUR"],
    "fruit_model": ["PICK_ENVELOPING", "PLACE_STABLE"],
    "kitchen_tool": ["PICK_VERTICAL", "PLACE_STABLE", "POUR"],
    "cleaning": ["PICK_VERTICAL", "PLACE_STABLE"],
    "tableware": ["PICK_VERTICAL", "PLACE_STABLE"],
    "tool": ["GRASP_PARALLEL_JAW", "PLACE_STABLE"],
    "block": ["PICK_VERTICAL", "PLACE_STABLE", "STACK"],
    "stationery": ["GRASP_PARALLEL_JAW", "PLACE_STABLE"],
}


# ---------------------------------------------------------------------------
# Material → fill colour for synthetic test images
# ---------------------------------------------------------------------------

_COLOUR_BY_MATERIAL: Dict[str, tuple] = {
    "plastic": (100, 180, 240),
    "metal": (180, 180, 190),
    "cardboard": (210, 170, 110),
    "ceramic": (230, 230, 220),
    "wood": (180, 140, 90),
    "cardboard_metal": (195, 175, 150),
    "plastic_metal": (140, 180, 215),
    "metal_plastic": (140, 180, 215),
}


# ---------------------------------------------------------------------------
# YCBLoader
# ---------------------------------------------------------------------------

class YCBLoader:
    """Load YCB ground-truth data and generate / download test images.

    Parameters
    ----------
    data_dir : str
        Root directory where per-object image directories are created.
    """

    _S3_BASE_URL = (
        "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/berkeley"
    )

    def __init__(
        self,
        data_dir: str = "/mnt/d/NeuralNet/golden-codex-robotics/data/ycb_objects",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info("YCBLoader initialised — data_dir: %s", self.data_dir)

    # -- download -----------------------------------------------------------

    def download_object_images(
        self, object_id: str, max_images: int = 10
    ) -> List[Path]:
        """Attempt to download RGB images for *object_id* from the YCB S3 bucket.

        The canonical URL pattern is::

            {S3_BASE}/{object_id}/{object_id}_berkeley_rgbd.tgz

        Because these tarballs are large (hundreds of MB), the method first
        tries a streaming download.  If the download fails or takes too long,
        it falls back to :meth:`generate_test_images` so that the rest of the
        pipeline can still be exercised.

        Returns
        -------
        list[Path]
            Paths to saved images under ``data_dir/{object_id}/``.
        """
        if object_id not in YCB_OBJECTS:
            logger.warning("Unknown object_id %r — not in YCB_OBJECTS", object_id)

        obj_dir = self.data_dir / object_id
        obj_dir.mkdir(parents=True, exist_ok=True)

        url = f"{self._S3_BASE_URL}/{object_id}/{object_id}_berkeley_rgbd.tgz"
        logger.info("Attempting download: %s", url)

        try:
            saved = self._download_and_extract(url, obj_dir, max_images)
            if saved:
                logger.info(
                    "Downloaded %d images for %s", len(saved), object_id
                )
                return saved
        except Exception as exc:
            logger.warning(
                "Download failed for %s (%s). Falling back to synthetic images.",
                object_id,
                exc,
            )

        # Fallback: generate synthetic placeholder images.
        return self.generate_test_images(object_id, count=max_images)

    @retry(
        retry=retry_if_exception_type((requests.RequestException, IOError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _download_and_extract(
        self, url: str, obj_dir: Path, max_images: int
    ) -> List[Path]:
        """Stream-download a ``.tgz`` archive and extract up to *max_images* RGB PNGs.

        Decorated with tenacity retry: 3 attempts, exponential backoff.
        """
        saved: List[Path] = []
        # Stream to avoid loading multi-hundred-MB tarballs into RAM.
        with requests.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            raw_bytes = io.BytesIO()
            total = int(resp.headers.get("content-length", 0))
            with tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=f"Downloading {obj_dir.name}",
                disable=total == 0,
            ) as pbar:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    raw_bytes.write(chunk)
                    pbar.update(len(chunk))

            raw_bytes.seek(0)
            with tarfile.open(fileobj=raw_bytes, mode="r:gz") as tar:
                count = 0
                for member in tar.getmembers():
                    if count >= max_images:
                        break
                    name_lower = member.name.lower()
                    if not (
                        name_lower.endswith(".png")
                        or name_lower.endswith(".jpg")
                        or name_lower.endswith(".jpeg")
                    ):
                        continue
                    # Only keep RGB images (skip depth maps).
                    if "depth" in name_lower or "mask" in name_lower:
                        continue

                    extracted = tar.extractfile(member)
                    if extracted is None:
                        continue

                    out_name = f"rgb_{count:04d}.png"
                    out_path = obj_dir / out_name
                    img = Image.open(extracted).convert("RGB")
                    img.save(out_path)
                    saved.append(out_path)
                    count += 1
                    logger.debug("Extracted %s -> %s", member.name, out_path)

        return saved

    # -- synthetic test images ----------------------------------------------

    def generate_test_images(
        self, object_id: str, count: int = 5
    ) -> List[Path]:
        """Generate simple synthetic test images for *object_id*.

        Each image is a solid-coloured rectangle whose aspect ratio mirrors the
        real object's dimensions, with the object name overlaid as text.  The
        fill colour is derived from the material type.

        These are **not** photorealistic images — they exist so that the full
        hash -> schema pipeline can be tested end-to-end without requiring a
        network download.

        Returns
        -------
        list[Path]
            Paths to saved PNG images under ``data_dir/{object_id}/``.
        """
        gt = YCB_OBJECTS.get(object_id)
        if gt is None:
            logger.error(
                "Cannot generate test images: %r not in YCB_OBJECTS", object_id
            )
            return []

        obj_dir = self.data_dir / object_id
        obj_dir.mkdir(parents=True, exist_ok=True)

        dims = gt["dimensions_m"]  # [w, d, h] in metres
        material = gt["material"]
        name = gt["name"]

        # Map real-world dimensions to pixel sizes (scale factor ~1000 px/m).
        px_w = max(int(dims[0] * 1000), 40)
        px_h = max(int(dims[2] * 1000), 40)

        base_colour = _COLOUR_BY_MATERIAL.get(material, (160, 160, 160))

        saved: List[Path] = []
        for i in range(count):
            # Slight per-image colour jitter to simulate viewpoint variation.
            jitter = (i * 7) % 30 - 15
            fill = tuple(max(0, min(255, c + jitter)) for c in base_colour)

            canvas_w = px_w + 80
            canvas_h = px_h + 80
            img = Image.new("RGB", (canvas_w, canvas_h), color=(240, 240, 240))
            draw = ImageDraw.Draw(img)

            # Draw the object rectangle centred on the canvas.
            x0 = (canvas_w - px_w) // 2
            y0 = (canvas_h - px_h) // 2
            draw.rectangle(
                [x0, y0, x0 + px_w, y0 + px_h], fill=fill, outline=(40, 40, 40)
            )

            # Overlay object name.  Use default font (no TTF dependency).
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
            except (IOError, OSError):
                font = ImageFont.load_default()

            label = f"{object_id}\n{name}"
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_x = max(2, (canvas_w - text_w) // 2)
            text_y = 4
            draw.text((text_x, text_y), label, fill=(20, 20, 20), font=font)

            out_path = obj_dir / f"test_{i:04d}.png"
            img.save(out_path)
            saved.append(out_path)

        logger.info(
            "Generated %d test images for %s in %s", len(saved), object_id, obj_dir
        )
        return saved

    # -- ground truth lookup ------------------------------------------------

    def get_ground_truth(self, object_id: str) -> Optional[dict]:
        """Return the YCB_OBJECTS entry for *object_id*, or ``None``."""
        gt = YCB_OBJECTS.get(object_id)
        if gt is None:
            logger.warning("No ground truth for object_id %r", object_id)
        return gt

    def get_all_objects(self) -> dict:
        """Return the full ``YCB_OBJECTS`` dictionary."""
        return dict(YCB_OBJECTS)

    # -- SKB seed builder ---------------------------------------------------

    def build_skb_seed(self, object_id: str) -> dict:
        """Build a partial SKB dictionary pre-populated with YCB ground-truth data.

        The returned dict maps to the schema's ``layer_2_semantic_topology``
        fields and includes:

        - Physical properties (mass, bounding box, material, friction).
        - Default action primitives based on object category.
        - A provenance tag marking the data source as ``ycb_ground_truth``.

        Returns
        -------
        dict
            SKB seed dictionary, or an empty dict if *object_id* is unknown.
        """
        gt = YCB_OBJECTS.get(object_id)
        if gt is None:
            logger.error(
                "Cannot build SKB seed: %r not in YCB_OBJECTS", object_id
            )
            return {}

        material = gt["material"]
        category = gt["category"]
        dims = gt["dimensions_m"]

        friction = _FRICTION_BY_MATERIAL.get(material, 0.4)
        action_primitives = _ACTION_PRIMITIVES_BY_CATEGORY.get(
            category, ["PICK_VERTICAL", "PLACE_STABLE"]
        )

        skb_seed: dict = {
            "object_id": object_id,
            "source": "ycb_ground_truth",
            "layer_2_semantic_topology": {
                "object_name": gt["name"],
                "category": category,
                "graspable": gt["graspable"],
                "physical_properties": {
                    "mass_kg": gt["mass_kg"],
                    "bounding_box_m": {
                        "x": dims[0],
                        "y": dims[1],
                        "z": dims[2],
                    },
                    "material": material,
                    "friction_coefficient": friction,
                },
                "action_primitives": action_primitives,
            },
            "confidence": 1.0,  # Ground truth — maximum confidence.
            "provenance": {
                "dataset": "YCB Object and Model Set",
                "url": "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/",
                "measurement_type": "laboratory",
            },
        }

        logger.debug("Built SKB seed for %s: mass=%.3f kg", object_id, gt["mass_kg"])
        return skb_seed

    # -- full test dataset generation ---------------------------------------

    def generate_test_dataset(
        self, num_objects: int = 20, images_per_object: int = 5
    ) -> dict:
        """Generate test images for up to *num_objects* YCB objects.

        Returns a manifest dictionary::

            {
                "<object_id>": {
                    "images": [Path, ...],
                    "ground_truth": { ... },
                    "skb_seed": { ... },
                },
                ...
            }

        Parameters
        ----------
        num_objects : int
            Maximum number of objects to include (capped at catalogue size).
        images_per_object : int
            Number of synthetic images to generate per object.
        """
        object_ids = list(YCB_OBJECTS.keys())[:num_objects]
        manifest: dict = {}

        logger.info(
            "Generating test dataset: %d objects, %d images each",
            len(object_ids),
            images_per_object,
        )

        for oid in tqdm(object_ids, desc="Generating test dataset"):
            images = self.generate_test_images(oid, count=images_per_object)
            manifest[oid] = {
                "images": images,
                "ground_truth": self.get_ground_truth(oid),
                "skb_seed": self.build_skb_seed(oid),
            }

        logger.info(
            "Test dataset complete: %d objects, %d total images",
            len(manifest),
            sum(len(v["images"]) for v in manifest.values()),
        )
        return manifest
