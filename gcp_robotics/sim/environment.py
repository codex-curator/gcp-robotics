"""
TabletopEnvironment — PyBullet tabletop simulation for PCO benchmark testing.

Sets up a standard pick-and-place workspace with a table, adjustable camera,
and methods to spawn/remove objects, render camera frames as PIL Images,
and create reproducible or randomised scenes.
"""

from __future__ import annotations

import logging
import math
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pybullet
import pybullet_data
from PIL import Image

logger = logging.getLogger(__name__)

# Built-in URDF catalogue (relative to pybullet_data path)
BUILTIN_OBJECTS: Dict[str, str] = {
    "cube": "cube.urdf",
    "duck": "duck_vhacd.urdf",
    "lego": "lego/lego.urdf",
    "mug": "objects/mug.urdf",
    "tray": "tray/traybox.urdf",
    "sphere": "sphere_small.urdf",
    "block": "block.urdf",
    "teddy": "teddy_large.urdf",
    "soccerball": "soccerball.urdf",
}

BUILTIN_ROBOTS: Dict[str, str] = {
    "kuka": "kuka_iiwa/model.urdf",
    "panda": "franka_panda/panda.urdf",
    "xarm6": "xarm/xarm6_robot.urdf",
}

# Standard table surface height from table.urdf
TABLE_SURFACE_Z = 0.625

# Default colours for procedural objects (RGBA)
PRESET_COLOURS = [
    (0.9, 0.2, 0.2, 1.0),  # red
    (0.2, 0.6, 0.9, 1.0),  # blue
    (0.2, 0.8, 0.3, 1.0),  # green
    (0.95, 0.85, 0.1, 1.0),  # yellow
    (0.7, 0.3, 0.8, 1.0),  # purple
    (0.95, 0.55, 0.1, 1.0),  # orange
]


class TabletopEnvironment:
    """PyBullet tabletop environment for PCO benchmark testing.

    Manages a physics world with a ground plane and table, provides methods
    to populate the scene with URDF objects or procedural primitives, and
    renders camera frames as PIL Images suitable for perceptual hashing.
    """

    def __init__(self, gui: bool = False, camera_distance: float = 1.0) -> None:
        """Initialise the PyBullet world.

        Parameters
        ----------
        gui : bool
            If True, open the PyBullet GUI window.  Default is headless
            (DIRECT mode) which is required for CI / benchmark runs.
        camera_distance : float
            Distance of the virtual camera from the table centre.
        """
        mode = pybullet.GUI if gui else pybullet.DIRECT
        self._physics_client = pybullet.connect(mode)
        pybullet.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self._physics_client
        )

        # Physics setup
        pybullet.setGravity(0, 0, -9.81, physicsClientId=self._physics_client)
        pybullet.setTimeStep(1.0 / 240.0, physicsClientId=self._physics_client)

        # Load ground plane and table
        self._plane_id = pybullet.loadURDF(
            "plane.urdf", physicsClientId=self._physics_client
        )
        self._table_id = pybullet.loadURDF(
            "table/table.urdf",
            basePosition=[0, 0, 0],
            baseOrientation=pybullet.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self._physics_client,
        )

        # Camera parameters
        self._cam_distance = camera_distance
        self._cam_yaw = 45.0
        self._cam_pitch = -35.0
        self._cam_target = [0.0, 0.0, TABLE_SURFACE_Z]

        # Object tracking: object_id -> {"label": str, "urdf": str}
        self._objects: Dict[int, dict] = {}

        logger.info(
            "TabletopEnvironment initialised (mode=%s, cam_dist=%.2f)",
            "GUI" if gui else "DIRECT",
            camera_distance,
        )

    # ------------------------------------------------------------------
    # Object management
    # ------------------------------------------------------------------

    def add_object(
        self,
        urdf_path: str,
        position: List[float],
        orientation: Optional[List[float]] = None,
        label: Optional[str] = None,
        color: Optional[Tuple[float, float, float, float]] = None,
        global_scaling: float = 1.0,
    ) -> int:
        """Place a URDF object in the scene.

        Parameters
        ----------
        urdf_path : str
            URDF filename (resolved relative to ``pybullet_data``) or
            absolute path.
        position : list of float
            [x, y, z] world position.
        orientation : list of float, optional
            Quaternion [x, y, z, w].  Defaults to identity.
        label : str, optional
            Human-readable label for this object.
        color : tuple, optional
            (r, g, b, a) colour override applied to all visual shapes.
        global_scaling : float
            Uniform scale factor applied to the URDF.

        Returns
        -------
        int
            PyBullet body ID for the newly added object.
        """
        if orientation is None:
            orientation = [0.0, 0.0, 0.0, 1.0]

        obj_id = pybullet.loadURDF(
            urdf_path,
            basePosition=position,
            baseOrientation=orientation,
            globalScaling=global_scaling,
            physicsClientId=self._physics_client,
        )

        if color is not None:
            pybullet.changeVisualShape(
                obj_id, -1, rgbaColor=color,
                physicsClientId=self._physics_client,
            )

        self._objects[obj_id] = {
            "label": label or urdf_path,
            "urdf": urdf_path,
        }

        logger.debug(
            "Added object id=%d label=%s at %s",
            obj_id, self._objects[obj_id]["label"], position,
        )
        return obj_id

    def add_primitive(
        self,
        shape: str,
        position: List[float],
        half_extents: Optional[List[float]] = None,
        radius: float = 0.025,
        height: float = 0.05,
        mass: float = 0.1,
        color: Tuple[float, float, float, float] = (0.8, 0.2, 0.2, 1.0),
        label: Optional[str] = None,
    ) -> int:
        """Create a coloured primitive shape (cube, sphere, or cylinder).

        Parameters
        ----------
        shape : str
            One of ``"cube"``, ``"sphere"``, ``"cylinder"``.
        position : list of float
            [x, y, z] world position.
        half_extents : list of float, optional
            Half-extents for cube. Defaults to [0.025, 0.025, 0.025].
        radius : float
            Radius for sphere or cylinder.
        height : float
            Height for cylinder.
        mass : float
            Object mass in kg.
        color : tuple
            (r, g, b, a) colour.
        label : str, optional
            Human-readable label.

        Returns
        -------
        int
            PyBullet body ID.
        """
        if half_extents is None:
            half_extents = [0.025, 0.025, 0.025]

        if shape == "cube":
            col_id = pybullet.createCollisionShape(
                pybullet.GEOM_BOX, halfExtents=half_extents,
                physicsClientId=self._physics_client,
            )
            vis_id = pybullet.createVisualShape(
                pybullet.GEOM_BOX, halfExtents=half_extents, rgbaColor=color,
                physicsClientId=self._physics_client,
            )
        elif shape == "sphere":
            col_id = pybullet.createCollisionShape(
                pybullet.GEOM_SPHERE, radius=radius,
                physicsClientId=self._physics_client,
            )
            vis_id = pybullet.createVisualShape(
                pybullet.GEOM_SPHERE, radius=radius, rgbaColor=color,
                physicsClientId=self._physics_client,
            )
        elif shape == "cylinder":
            col_id = pybullet.createCollisionShape(
                pybullet.GEOM_CYLINDER, radius=radius, height=height,
                physicsClientId=self._physics_client,
            )
            vis_id = pybullet.createVisualShape(
                pybullet.GEOM_CYLINDER, radius=radius, length=height,
                rgbaColor=color,
                physicsClientId=self._physics_client,
            )
        else:
            raise ValueError(f"Unknown primitive shape: {shape!r}")

        obj_id = pybullet.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=position,
            physicsClientId=self._physics_client,
        )

        self._objects[obj_id] = {
            "label": label or f"{shape}_{obj_id}",
            "urdf": f"primitive:{shape}",
        }

        logger.debug(
            "Added primitive id=%d shape=%s label=%s at %s",
            obj_id, shape, self._objects[obj_id]["label"], position,
        )
        return obj_id

    def remove_object(self, object_id: int) -> None:
        """Remove an object from the scene."""
        if object_id in self._objects:
            pybullet.removeBody(object_id, physicsClientId=self._physics_client)
            del self._objects[object_id]
            logger.debug("Removed object id=%d", object_id)
        else:
            logger.warning("remove_object: id=%d not tracked", object_id)

    def get_object_pose(self, object_id: int) -> Tuple[List[float], List[float]]:
        """Return (position, orientation) for an object.

        Returns
        -------
        tuple
            ([x, y, z], [qx, qy, qz, qw])
        """
        pos, orn = pybullet.getBasePositionAndOrientation(
            object_id, physicsClientId=self._physics_client
        )
        return list(pos), list(orn)

    def set_object_pose(
        self,
        object_id: int,
        position: List[float],
        orientation: Optional[List[float]] = None,
    ) -> None:
        """Move an object to a new pose."""
        if orientation is None:
            _, orientation = self.get_object_pose(object_id)
        pybullet.resetBasePositionAndOrientation(
            object_id, position, orientation,
            physicsClientId=self._physics_client,
        )

    @property
    def objects(self) -> Dict[int, dict]:
        """Return a copy of the tracked objects dictionary."""
        return dict(self._objects)

    # ------------------------------------------------------------------
    # Camera and rendering
    # ------------------------------------------------------------------

    def set_camera(
        self,
        distance: Optional[float] = None,
        yaw: Optional[float] = None,
        pitch: Optional[float] = None,
        target: Optional[List[float]] = None,
    ) -> None:
        """Update camera parameters for subsequent renders."""
        if distance is not None:
            self._cam_distance = distance
        if yaw is not None:
            self._cam_yaw = yaw
        if pitch is not None:
            self._cam_pitch = pitch
        if target is not None:
            self._cam_target = list(target)

    def render_camera(self, width: int = 224, height: int = 224) -> Image.Image:
        """Render the current scene and return a PIL RGB Image.

        Uses ``pybullet.getCameraImage`` with an OpenGL renderer in DIRECT
        mode.  The view matrix is computed from the stored camera
        parameters (distance, yaw, pitch, target).
        """
        view_matrix = pybullet.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self._cam_target,
            distance=self._cam_distance,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2,
        )

        aspect = width / height
        proj_matrix = pybullet.computeProjectionMatrixFOV(
            fov=60.0,
            aspect=aspect,
            nearVal=0.01,
            farVal=10.0,
        )

        _, _, rgba, _, _ = pybullet.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=pybullet.ER_TINY_RENDERER,
            physicsClientId=self._physics_client,
        )

        rgba_array = np.array(rgba, dtype=np.uint8).reshape(height, width, 4)
        rgb_array = rgba_array[:, :, :3]
        return Image.fromarray(rgb_array, mode="RGB")

    # ------------------------------------------------------------------
    # Physics stepping
    # ------------------------------------------------------------------

    def step(self, n: int = 1) -> None:
        """Advance the physics simulation by *n* steps."""
        for _ in range(n):
            pybullet.stepSimulation(physicsClientId=self._physics_client)

    # ------------------------------------------------------------------
    # Scene factories
    # ------------------------------------------------------------------

    def create_pick_and_place_scene(self, seed: Optional[int] = None) -> Dict[str, int]:
        """Create a standard pick-and-place scene with 3-5 random objects.

        Objects are placed on the table surface with enough spacing to
        avoid overlaps.  A random subset of ``BUILTIN_OBJECTS`` is chosen
        and each is assigned a random colour.

        Parameters
        ----------
        seed : int, optional
            RNG seed for reproducibility.

        Returns
        -------
        dict
            Mapping of ``{label: object_id}``.
        """
        rng = random.Random(seed)
        n_objects = rng.randint(3, 5)

        # Pick random objects from the catalogue
        available = list(BUILTIN_OBJECTS.items())
        chosen = rng.sample(available, min(n_objects, len(available)))

        # Generate non-overlapping positions on the table surface.
        # Table centre is at (0, 0), surface at TABLE_SURFACE_Z.
        # Workspace radius ~ 0.25 m.
        positions = self._generate_spread_positions(
            n=len(chosen), workspace_radius=0.25, min_spacing=0.12, rng=rng,
        )

        result: Dict[str, int] = {}
        for i, (label, urdf) in enumerate(chosen):
            colour = PRESET_COLOURS[i % len(PRESET_COLOURS)]
            x, y = positions[i]
            z = TABLE_SURFACE_Z + 0.02  # small offset above surface

            obj_id = self.add_object(
                urdf_path=urdf,
                position=[x, y, z],
                label=label,
                color=colour,
            )
            result[label] = obj_id

        # Let objects settle
        self.step(n=120)

        logger.info(
            "Created pick-and-place scene with %d objects: %s",
            len(result), list(result.keys()),
        )
        return result

    def create_scene_from_config(self, config: dict) -> Dict[str, int]:
        """Create a scene from a configuration dictionary.

        Expected format::

            {
                "objects": [
                    {
                        "urdf": "cube.urdf",
                        "position": [0.1, 0.0, 0.65],
                        "orientation": [0, 0, 0, 1],  # optional
                        "label": "red_cube",            # optional
                        "color": [0.9, 0.2, 0.2, 1.0], # optional
                        "scale": 1.0,                   # optional
                    },
                    ...
                ]
            }

        Returns
        -------
        dict
            Mapping of ``{label: object_id}``.
        """
        result: Dict[str, int] = {}
        for obj_spec in config.get("objects", []):
            urdf = obj_spec["urdf"]
            position = obj_spec["position"]
            orientation = obj_spec.get("orientation")
            label = obj_spec.get("label", urdf)
            color = obj_spec.get("color")
            if color is not None:
                color = tuple(color)
            scale = obj_spec.get("scale", 1.0)

            obj_id = self.add_object(
                urdf_path=urdf,
                position=position,
                orientation=orientation,
                label=label,
                color=color,
                global_scaling=scale,
            )
            result[label] = obj_id

        # Let objects settle
        self.step(n=120)

        logger.info(
            "Created scene from config with %d objects: %s",
            len(result), list(result.keys()),
        )
        return result

    def clear_objects(self) -> None:
        """Remove all spawned objects (keeps table and plane)."""
        for obj_id in list(self._objects.keys()):
            pybullet.removeBody(obj_id, physicsClientId=self._physics_client)
        self._objects.clear()
        logger.debug("All objects cleared")

    # ------------------------------------------------------------------
    # Context manager and cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Disconnect from PyBullet."""
        try:
            pybullet.disconnect(physicsClientId=self._physics_client)
        except pybullet.error:
            pass  # already disconnected
        logger.info("TabletopEnvironment closed")

    def __enter__(self) -> "TabletopEnvironment":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_spread_positions(
        n: int,
        workspace_radius: float,
        min_spacing: float,
        rng: random.Random,
        max_attempts: int = 200,
    ) -> List[Tuple[float, float]]:
        """Generate *n* 2-D positions with minimum spacing via rejection sampling."""
        positions: List[Tuple[float, float]] = []
        for _ in range(n):
            for _attempt in range(max_attempts):
                x = rng.uniform(-workspace_radius, workspace_radius)
                y = rng.uniform(-workspace_radius, workspace_radius)
                if all(
                    math.hypot(x - px, y - py) >= min_spacing
                    for px, py in positions
                ):
                    positions.append((x, y))
                    break
            else:
                # Fallback: place with slight random offset from centre
                angle = 2 * math.pi * len(positions) / n
                r = workspace_radius * 0.6
                positions.append((r * math.cos(angle), r * math.sin(angle)))
        return positions
