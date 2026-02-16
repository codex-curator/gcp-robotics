"""
CompositeHasher — perceptual hash computation engine for the Golden Codex Protocol.

Computes multiple perceptual hash representations of images for O(1) Fast Path
lookup in the CodexRegistry. Each hash captures a different perceptual axis:

  - dHash (64-bit):   Gradient-based difference hash. Primary lookup key.
  - pHash (64-bit):   DCT-based perceptual hash. Secondary index for cross-validation.
  - pHash-256 (256b): High-fidelity perceptual hash used as the PPP (Persistent
                       Provenance Pointer) for global registry resolution.
  - Color Hash:       Mean RGB per 4x4 grid cell. Distinguishes geometrically
                       identical but chromatically distinct objects.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Union

import imagehash
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompositeHash:
    """All perceptual hash representations for a single image."""

    dhash_hex: str
    phash_hex: str
    ppp_256bit: str
    color_hash: str

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _open_image(image_path_or_pil: Union[str, Path, Image.Image]) -> Image.Image:
    """Normalise input to a PIL Image, opening from disk if necessary."""
    if isinstance(image_path_or_pil, Image.Image):
        return image_path_or_pil
    path = Path(image_path_or_pil)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def _imagehash_to_hex(ih: imagehash.ImageHash) -> str:
    """Convert an imagehash.ImageHash to a '0x…' prefixed hex string."""
    return f"0x{str(ih).upper()}"


# ---------------------------------------------------------------------------
# CompositeHasher
# ---------------------------------------------------------------------------

class CompositeHasher:
    """Computes composite perceptual hashes of images.

    All ``compute_*`` methods accept either a filesystem path (str / Path)
    or an already-opened ``PIL.Image.Image``.
    """

    # -- dHash (64-bit, gradient-based) ------------------------------------

    def compute_dhash(
        self, image_path_or_pil: Union[str, Path, Image.Image]
    ) -> str:
        """Return 64-bit dHash as a ``0x``-prefixed hex string.

        Uses ``imagehash.dhash`` with ``hash_size=8`` (8x8 = 64 bits).
        """
        img = _open_image(image_path_or_pil)
        ih = imagehash.dhash(img, hash_size=8)
        hex_str = _imagehash_to_hex(ih)
        logger.debug("dHash computed: %s", hex_str)
        return hex_str

    # -- pHash (64-bit, DCT-based) -----------------------------------------

    def compute_phash(
        self, image_path_or_pil: Union[str, Path, Image.Image], hash_size: int = 8
    ) -> str:
        """Return pHash as a ``0x``-prefixed hex string.

        Default ``hash_size=8`` yields a 64-bit hash.
        """
        img = _open_image(image_path_or_pil)
        ih = imagehash.phash(img, hash_size=hash_size)
        hex_str = _imagehash_to_hex(ih)
        logger.debug("pHash (size=%d) computed: %s", hash_size, hex_str)
        return hex_str

    # -- pHash-256 (PPP — Persistent Provenance Pointer) -------------------

    def compute_phash_256(
        self, image_path_or_pil: Union[str, Path, Image.Image]
    ) -> str:
        """Return 256-bit pHash as a ``0x``-prefixed hex string (hash_size=16).

        This is the PPP (Persistent Provenance Pointer) used for global
        registry resolution across distributed Codex nodes.
        """
        return self.compute_phash(image_path_or_pil, hash_size=16)

    # -- Color Hash (4x4 grid mean-RGB) ------------------------------------

    def compute_color_hash(
        self, image_path_or_pil: Union[str, Path, Image.Image]
    ) -> str:
        """Return a colour-moment hash string.

        The image is divided into a 4x4 grid. For each of the 16 cells the
        mean R, G, B values are computed, quantised to a single byte each,
        and concatenated into a 48-byte (96 hex-char) string prefixed with
        ``0x``.  This distinguishes geometrically identical but chromatically
        distinct objects (e.g. a red mug vs. a blue mug with the same shape).
        """
        img = _open_image(image_path_or_pil)
        arr = np.asarray(img)  # shape (H, W, 3)
        h, w, _ = arr.shape

        cell_h = h // 4
        cell_w = w // 4

        hex_parts: list[str] = []
        for row in range(4):
            for col in range(4):
                r0 = row * cell_h
                r1 = (row + 1) * cell_h if row < 3 else h
                c0 = col * cell_w
                c1 = (col + 1) * cell_w if col < 3 else w

                cell = arr[r0:r1, c0:c1]
                mean_r = int(np.mean(cell[:, :, 0]))
                mean_g = int(np.mean(cell[:, :, 1]))
                mean_b = int(np.mean(cell[:, :, 2]))
                hex_parts.append(f"{mean_r:02X}{mean_g:02X}{mean_b:02X}")

        color_hex = "0x" + "".join(hex_parts)
        logger.debug("Color hash computed: %s", color_hex)
        return color_hex

    # -- Composite (all four hashes) ---------------------------------------

    def compute_composite(
        self, image_path_or_pil: Union[str, Path, Image.Image]
    ) -> CompositeHash:
        """Compute all four hashes in one pass and return a ``CompositeHash``.

        Opens the image once and reuses the PIL object for each sub-hash.
        """
        img = _open_image(image_path_or_pil)
        composite = CompositeHash(
            dhash_hex=self.compute_dhash(img),
            phash_hex=self.compute_phash(img),
            ppp_256bit=self.compute_phash_256(img),
            color_hash=self.compute_color_hash(img),
        )
        logger.info("Composite hash computed: dHash=%s", composite.dhash_hex)
        return composite

    # -- Hamming distance --------------------------------------------------

    @staticmethod
    def hamming_distance(hash1: str, hash2: str) -> int:
        """Bitwise Hamming distance between two ``0x``-prefixed hex strings.

        Both hashes must encode the same number of bits.
        """
        # Strip optional "0x" / "0X" prefix
        h1 = hash1.lower().removeprefix("0x")
        h2 = hash2.lower().removeprefix("0x")

        if len(h1) != len(h2):
            raise ValueError(
                f"Hash length mismatch: {len(h1)} hex chars vs {len(h2)} hex chars"
            )

        xor_val = int(h1, 16) ^ int(h2, 16)
        return bin(xor_val).count("1")

    # -- Content hash (SHA-256 of raw bytes) -------------------------------

    @staticmethod
    def content_hash_sha256(image_path: Union[str, Path]) -> str:
        """Return the SHA-256 hex digest of the raw file bytes."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        logger.debug("SHA-256: %s for %s", sha, path)
        return sha
