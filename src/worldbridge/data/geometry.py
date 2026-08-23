"""Camera, pointmap, rigid trajectory and visibility geometry for MOVi-F."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .movif import MOViSample


@dataclass(frozen=True)
class CameraModel:
    height: int
    width: int
    focal_length: float
    sensor_width: float
    quaternion_order: str = "wxyz"
    forward_axis: str = "-z"
    depth_is_euclidean: bool = True

    @property
    def fx(self) -> float:
        return self.focal_length / self.sensor_width * self.width

    @property
    def fy(self) -> float:
        return self.fx

    @property
    def cx(self) -> float:
        return (self.width - 1) / 2.0

    @property
    def cy(self) -> float:
        return (self.height - 1) / 2.0

    @staticmethod
    def quaternion_matrix(quaternions: np.ndarray, order: str = "wxyz") -> np.ndarray:
        q = np.asarray(quaternions, dtype=np.float64)
        if order == "wxyz":
            w, x, y, z = [q[..., i] for i in range(4)]
        elif order == "xyzw":
            x, y, z, w = [q[..., i] for i in range(4)]
        else:
            raise ValueError(f"Unsupported quaternion order: {order}")
        n = np.sqrt(w*w + x*x + y*y + z*z).clip(1e-12)
        w, x, y, z = w/n, x/n, y/n, z/n
        R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
        R[..., 0, 0] = 1 - 2*(y*y + z*z)
        R[..., 0, 1] = 2*(x*y - z*w)
        R[..., 0, 2] = 2*(x*z + y*w)
        R[..., 1, 0] = 2*(x*y + z*w)
        R[..., 1, 1] = 1 - 2*(x*x + z*z)
        R[..., 1, 2] = 2*(y*z - x*w)
        R[..., 2, 0] = 2*(x*z - y*w)
        R[..., 2, 1] = 2*(y*z + x*w)
        R[..., 2, 2] = 1 - 2*(x*x + y*y)
        return R

    def camera_to_world(self, camera_points: np.ndarray, positions: np.ndarray, quaternions: np.ndarray) -> np.ndarray:
        R = self.quaternion_matrix(quaternions, self.quaternion_order)
        return np.einsum("...j,...ij->...i", camera_points, R) + np.asarray(positions)

    def world_to_camera(self, world_points: np.ndarray, positions: np.ndarray, quaternions: np.ndarray) -> np.ndarray:
        R = self.quaternion_matrix(quaternions, self.quaternion_order)
        return np.einsum("...j,...ji->...i", np.asarray(world_points) - np.asarray(positions), R)

    def backproject(self, depth: np.ndarray, positions: np.ndarray, quaternions: np.ndarray) -> np.ndarray:
        """Backproject radial MOVi depth into Kubric world coordinates.

        Kubric's camera looks down local -Z; image v grows down while local Y
        grows up. Native ``depth`` is linearly decoded uint16 camera distance,
        so the ray is normalized before applying that distance.
        """
        d = np.asarray(depth, dtype=np.float64)
        vv, uu = np.meshgrid(np.arange(self.height), np.arange(self.width), indexing="ij")
        x = (uu - self.cx) / self.fx
        y = -(vv - self.cy) / self.fy
        ray = np.stack([x, y, -np.ones_like(x)], axis=-1)
        if self.depth_is_euclidean:
            ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
        local = ray * d[..., None]
        return self.camera_to_world(local, np.asarray(positions), np.asarray(quaternions))

    def backproject_pixels(self, depth: np.ndarray, uv: np.ndarray, positions: np.ndarray, quaternions: np.ndarray) -> np.ndarray:
        d = np.asarray(depth, dtype=np.float64).reshape(-1)
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        x = (uv[:, 0] - self.cx) / self.fx
        y = -(uv[:, 1] - self.cy) / self.fy
        ray = np.stack([x, y, -np.ones_like(x)], axis=-1)
        if self.depth_is_euclidean:
            ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
        return self.camera_to_world(ray * d[:, None], positions, quaternions)

    def project(self, world_points: np.ndarray, positions: np.ndarray, quaternions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cam = self.world_to_camera(world_points, positions, quaternions)
        view_depth = -cam[..., 2]
        safe = np.where(view_depth > 1e-8, view_depth, 1.0)
        u = self.fx * cam[..., 0] / safe + self.cx
        v = self.cy - self.fy * cam[..., 1] / safe
        return np.stack([u, v], axis=-1), view_depth, np.linalg.norm(cam, axis=-1)


class GeometryBuilder:
    """On-demand geometry; no full ``[source,target,H,W,3]`` cache is kept."""

    def __init__(self, sample: MOViSample, depth_tolerance: float = 0.05, depth_relative_tolerance: float = 0.01):
        self.sample = sample
        self.camera = CameraModel(sample.height, sample.width, sample.focal_length, sample.sensor_width)
        self.depth_tolerance = float(depth_tolerance)
        self.depth_relative_tolerance = float(depth_relative_tolerance)
        self._camera_rot = self.camera.quaternion_matrix(sample.camera_quaternions)
        self._object_rot = self.camera.quaternion_matrix(sample.instance_quaternions)
        self._world_points: np.ndarray | None = None

    @staticmethod
    def _check_coordinate_frame(coordinate_frame: str) -> str:
        coordinate_frame = str(coordinate_frame).lower()
        if coordinate_frame not in {"anchor", "source"}:
            raise ValueError(f"coordinate_frame must be 'anchor' or 'source', got {coordinate_frame!r}")
        return coordinate_frame

    def world_to_camera_frame(self, world_points: np.ndarray, frame: int) -> np.ndarray:
        """Express world points in the camera coordinate system at ``frame``."""
        frame = int(frame)
        if not 0 <= frame < self.sample.num_frames:
            raise ValueError(f"frame={frame} outside [0,{self.sample.num_frames})")
        return np.einsum(
            "...j,ji->...i", np.asarray(world_points) - self.sample.camera_positions[frame],
            self._camera_rot[frame],
        )

    def camera_frame_to_world(self, camera_points: np.ndarray, frame: int) -> np.ndarray:
        """Convert points in camera ``frame`` coordinates back to world."""
        frame = int(frame)
        if not 0 <= frame < self.sample.num_frames:
            raise ValueError(f"frame={frame} outside [0,{self.sample.num_frames})")
        return np.einsum("...j,ij->...i", np.asarray(camera_points), self._camera_rot[frame]) \
            + self.sample.camera_positions[frame]

    def pointmaps(self, coordinate_frame: str = "anchor") -> tuple[np.ndarray, np.ndarray]:
        """Return diagonal pointmaps in a common or per-frame camera frame.

        ``anchor`` preserves the historical clip-frame-0 convention.  ``source``
        expresses ``P[t]`` in camera ``t`` coordinates, which is the diagonal
        special case of source-conditioned local pointmaps.
        """
        coordinate_frame = self._check_coordinate_frame(coordinate_frame)
        if self._world_points is None:
            self._world_points = np.stack([
                self.camera.backproject(self.sample.depth[t], self.sample.camera_positions[t], self.sample.camera_quaternions[t])
                for t in range(self.sample.num_frames)
            ], axis=0)
        if coordinate_frame == "anchor":
            p = self.world_to_anchor(self._world_points)
        else:
            p = np.einsum(
                "t...j,tji->t...i", self._world_points - self.sample.camera_positions[:, None, None, :],
                self._camera_rot,
            )
        return p.astype(np.float32), self.sample.depth_valid.copy()

    def world_to_anchor(self, world_points: np.ndarray) -> np.ndarray:
        return self.world_to_camera_frame(world_points, 0)

    def anchor_to_world(self, anchor_points: np.ndarray) -> np.ndarray:
        return self.camera_frame_to_world(anchor_points, 0)

    def _source_world(self, source: int, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        uv = np.asarray(uv, dtype=np.int64)
        u = uv[:, 0].clip(0, self.sample.width - 1)
        v = uv[:, 1].clip(0, self.sample.height - 1)
        source_world = self.camera.backproject_pixels(
            self.sample.depth[source, v, u], np.stack([u, v], axis=-1),
            self.sample.camera_positions[source], self.sample.camera_quaternions[source])
        instance = self.sample.segmentation[source, v, u]
        return source_world, instance, self.sample.depth_valid[source, v, u]

    def _visible(self, world_target: np.ndarray, target: int, instance: np.ndarray) -> np.ndarray:
        uv, view_depth, radial_depth = self.camera.project(
            world_target, self.sample.camera_positions[target], self.sample.camera_quaternions[target])
        u = np.floor(uv[:, 0] + 0.5).astype(np.int64)
        v = np.floor(uv[:, 1] + 0.5).astype(np.int64)
        inside = ((view_depth > 0) & (u >= 0) & (u < self.sample.width) & (v >= 0) & (v < self.sample.height))
        out = np.zeros(len(instance), dtype=bool)
        if not np.any(inside):
            return out
        ii = np.flatnonzero(inside)
        observed_seg = self.sample.segmentation[target, v[ii], u[ii]]
        same_instance = observed_seg == instance[ii]
        target_depth = self.sample.depth[target, v[ii], u[ii]].astype(np.float64)
        tol = self.depth_tolerance + self.depth_relative_tolerance * np.maximum(target_depth, 1.0)
        depth_ok = np.isfinite(target_depth) & self.sample.depth_valid[target, v[ii], u[ii]] & (np.abs(target_depth - radial_depth[ii]) <= tol)
        out[ii] = same_instance & depth_ok
        return out

    def trajectory(self, source: int, uv: np.ndarray, coordinate_frame: str = "anchor",
                   compute_visibility: bool = True
                   ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """Compute X[source,target,p], optional M, and V for integer pixels.

        The source-grid semantics and validity masks are identical for both
        coordinate conventions; only the XYZ basis changes.  ``source`` mode
        applies one fixed SE(3) transform (camera ``source``) to every target
        point in the trajectory. Training routes that supervise only XYZ may
        disable the comparatively expensive target-camera visibility audit.
        """
        coordinate_frame = self._check_coordinate_frame(coordinate_frame)
        source = int(source)
        source_world, instance, source_valid = self._source_world(source, uv)
        n = len(instance); T = self.sample.num_frames
        x_world = np.repeat(source_world[:, None, :], T, axis=1)
        v_valid = np.repeat(source_valid[:, None], T, axis=1)
        for k in np.unique(instance):
            if k <= 0 or k > self.sample.num_instances:
                continue
            mask = instance == k
            obj = int(k - 1)
            # Static background and object state arrays are distinct: validity is
            # not visibility. A valid rigid state still yields X when occluded.
            local = (source_world[mask] - self.sample.instance_positions[obj, source]) @ self._object_rot[obj, source]
            x_world[mask] = np.einsum("nj,tij->nti", local, self._object_rot[obj]) + self.sample.instance_positions[obj]
            state_valid = np.isfinite(self.sample.instance_positions[obj]).all(axis=-1) & np.isfinite(self._object_rot[obj]).all(axis=(1,2))
            v_valid[mask] &= state_valid[None, :]
        x_coordinates = (
            self.world_to_anchor(x_world)
            if coordinate_frame == "anchor"
            else self.world_to_camera_frame(x_world, source)
        )
        visible = (
            np.stack([self._visible(x_world[:, t], t, instance) for t in range(T)], axis=1)
            if compute_visibility else None
        )
        return x_coordinates.astype(np.float32), visible, v_valid

    def trajectory_block(self, source: int, start: int = 0, end: int | None = None,
                         coordinate_frame: str = "anchor", compute_visibility: bool = True
                         ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
        end = self.sample.height * self.sample.width if end is None else int(end)
        flat = np.arange(start, end, dtype=np.int64)
        uv = np.stack([flat % self.sample.width, flat // self.sample.width], axis=-1)
        x, m, v = self.trajectory(
            source, uv, coordinate_frame=coordinate_frame, compute_visibility=compute_visibility,
        )
        return x, m, v, uv

    def query(self, source: np.ndarray, uv: np.ndarray, coordinate_frame: str = "anchor") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        coordinate_frame = self._check_coordinate_frame(coordinate_frame)
        source = np.asarray(source, dtype=np.int64).reshape(-1)
        uv = np.asarray(uv, dtype=np.int64).reshape(-1, 2)
        if len(source) != len(uv):
            raise ValueError("source and uv lengths differ")
        xs = []; ms = []; vs = []
        for s in np.unique(source):
            idx = np.flatnonzero(source == s)
            x, m, v = self.trajectory(int(s), uv[idx], coordinate_frame=coordinate_frame)
            xs.append((idx, x)); ms.append((idx, m)); vs.append((idx, v))
        shape = (len(source), self.sample.num_frames)
        xout = np.empty(shape + (3,), np.float32); mout = np.empty(shape, bool); vout = np.empty(shape, bool)
        for idx, x in xs: xout[idx] = x
        for idx, m in ms: mout[idx] = m
        for idx, v in vs: vout[idx] = v
        return xout, mout, vout

    def project_trajectory(self, x_coordinates: np.ndarray, target: int,
                           coordinate_frame: str = "anchor") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        coordinate_frame = self._check_coordinate_frame(coordinate_frame)
        if coordinate_frame == "anchor":
            world = self.anchor_to_world(x_coordinates)
        else:
            raise ValueError("source-frame trajectories require the source index for inverse conversion")
        return self.camera.project(world, self.sample.camera_positions[target], self.sample.camera_quaternions[target])

    def source_to_anchor(self, source_points: np.ndarray, source: int) -> np.ndarray:
        """Convert source-camera coordinates to the historical frame-0 anchor."""
        world = self.camera_frame_to_world(source_points, int(source))
        return self.world_to_anchor(world)

    def anchor_to_source(self, anchor_points: np.ndarray, source: int) -> np.ndarray:
        """Convert frame-0 anchor coordinates to camera ``source`` coordinates."""
        world = self.anchor_to_world(anchor_points)
        return self.world_to_camera_frame(world, int(source))
