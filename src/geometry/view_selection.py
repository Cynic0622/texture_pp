"""Per-island view selection."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


__all__ = ["select_views"]


# ===========================================================================
# ===========================================================================

@dataclass
class _MeshCache:
    centers: np.ndarray
    normals: np.ndarray
    areas: np.ndarray
    valid: np.ndarray
    fv: np.ndarray

    @classmethod
    def build(cls, mesh) -> "_MeshCache":
        F = len(mesh.faces)
        fv = np.zeros((F, 3), dtype=np.int64)
        has_tri = np.zeros(F, dtype=bool)
        for fi, (face_verts, _, _) in enumerate(mesh.faces):
            if len(face_verts) >= 3:
                fv[fi, 0] = face_verts[0]
                fv[fi, 1] = face_verts[1]
                fv[fi, 2] = face_verts[2]
                has_tri[fi] = True

        verts = np.asarray(mesh.vertices, dtype=np.float64)
        v0 = verts[fv[:, 0]]
        v1 = verts[fv[:, 1]]
        v2 = verts[fv[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        cn = np.linalg.norm(cross, axis=1)
        nonzero = cn > 1e-14
        valid = has_tri & nonzero

        safe_cn = np.where(nonzero, cn, 1.0)
        normals = cross / safe_cn[:, None]
        areas = cn * 0.5
        centers = (v0 + v1 + v2) / 3.0

        normals[~valid] = 0.0
        centers[~valid] = 0.0
        areas[~valid] = 0.0
        return cls(centers=centers, normals=normals, areas=areas,
                   valid=valid, fv=fv)


# ===========================================================================
# ===========================================================================

def _compute_position_welding(positions: np.ndarray, eps: float) -> Optional[np.ndarray]:
    if eps <= 0:
        return None
    pos = np.asarray(positions, dtype=np.float64)
    if pos.size == 0 or pos.ndim < 2:
        return None
    rounded = np.round(pos / eps).astype(np.int64)
    _, inverse = np.unique(rounded, axis=0, return_inverse=True)
    return inverse.astype(np.int64)


# ===========================================================================
# ===========================================================================

def _compute_xatlas_uv_per_face(mesh,
                                canonical_v: Optional[np.ndarray] = None,
                                log: bool = False,
                                ) -> Optional[List[List[int]]]:
    try:
        import xatlas
    except ImportError:
        if log:
            print("[view_selection] xatlas not installed — "
                  "use_xatlas_charts ignored. pip install xatlas")
        return None

    if len(mesh.faces) == 0:
        return None

    raw_verts = np.asarray(mesh.vertices, dtype=np.float32)
    raw_face_idx = np.array(
        [list(f[0][:3]) for f in mesh.faces], dtype=np.int64,
    )

    if canonical_v is not None and len(canonical_v) == len(raw_verts):
        n_unique = int(canonical_v.max()) + 1
        welded_verts = np.zeros((n_unique, 3), dtype=np.float32)
        welded_verts[canonical_v] = raw_verts
        welded_face_idx = canonical_v[raw_face_idx].astype(np.uint32)
        nondegenerate = ~(
            (welded_face_idx[:, 0] == welded_face_idx[:, 1])
            | (welded_face_idx[:, 1] == welded_face_idx[:, 2])
            | (welded_face_idx[:, 0] == welded_face_idx[:, 2])
        )
        atlas_verts = welded_verts
        atlas_faces = welded_face_idx[nondegenerate]
        kept_face_indices = np.nonzero(nondegenerate)[0]
    else:
        atlas_verts = raw_verts
        atlas_faces = raw_face_idx.astype(np.uint32)
        kept_face_indices = np.arange(len(mesh.faces), dtype=np.int64)

    atlas = xatlas.Atlas()
    atlas.add_mesh(atlas_verts, atlas_faces)
    atlas.generate()
    _vmapping, indices, _uvs = atlas[0]

    out: List[List[int]] = [list(mesh.faces[fi][1]) for fi in range(len(mesh.faces))]
    for kept_pos, orig_fi in enumerate(kept_face_indices):
        out[int(orig_fi)] = [int(x) for x in indices[kept_pos]]

    if log:
        n_xatlas_uv = int(indices.max()) + 1 if indices.size > 0 else 0
        n_atlas_verts = len(atlas_verts)
        weld_note = (
            f"weld {len(raw_verts)}->{n_atlas_verts}"
            if canonical_v is not None and n_atlas_verts < len(raw_verts)
            else "no welding"
        )
        n_dropped_degen = len(raw_face_idx) - len(atlas_faces)
        print(f"[view_selection] xatlas charts: input verts={n_atlas_verts} "
              f"({weld_note}), kept faces={len(atlas_faces)} "
              f"(dropped {n_dropped_degen} degenerate after welding), "
              f"original UV slots {len(getattr(mesh, 'uvs', []))} -> "
              f"xatlas UV slots {n_xatlas_uv} (used only for island partitioning)")
    return out


# ===========================================================================
# ===========================================================================

@dataclass
class ViewSpec:
    view: np.ndarray
    persp: dict
    cam_pos: np.ndarray
    forward: np.ndarray
    center: np.ndarray
    normal: np.ndarray
    faces: List[int] = field(default_factory=list)
    source: str = "island"

    def to_dict(self) -> dict:
        right = self.view[0, :3]
        up = self.view[1, :3]
        return {
            "view": self.view.astype(np.float32),
            "persp": {k: (float(v) if isinstance(v, (int, float)) else v)
                      for k, v in self.persp.items()},
            "center": self.center.astype(np.float32),
            "normal": self.normal.astype(np.float32),
            "axes": {
                "right": right.astype(np.float32),
                "up": up.astype(np.float32),
                "forward": self.forward.astype(np.float32),
            },
            "faces": list(self.faces),
            "_source": self.source,
        }


# ===========================================================================
# ===========================================================================

def _detect_uv_islands(mesh,
                       canonical_v: Optional[np.ndarray] = None,
                       canonical_uv: Optional[np.ndarray] = None,
                       faces_uvs_override: Optional[List[List[int]]] = None,
                       ) -> Tuple[List[List[int]], List[int]]:
    del canonical_v
    no_uv: List[int] = []
    edge_to_faces: dict = defaultdict(list)
    face_edges: dict = {}

    def _uv_key(uv_idx: int) -> int:
        return int(canonical_uv[uv_idx]) if canonical_uv is not None else int(uv_idx)

    for fi, (verts, orig_uvs, _) in enumerate(mesh.faces):
        uvs = (faces_uvs_override[fi]
               if faces_uvs_override is not None else orig_uvs)
        if len(verts) != len(uvs):
            no_uv.append(fi)
            continue
        edges = set()
        n = len(uvs)
        for i in range(n):
            k1 = _uv_key(uvs[i])
            k2 = _uv_key(uvs[(i + 1) % n])
            edges.add(tuple(sorted((k1, k2))))
        face_edges[fi] = edges
        for e in edges:
            edge_to_faces[e].append(fi)

    visited: set = set()
    islands: List[List[int]] = []
    for start in face_edges:
        if start in visited:
            continue
        comp: List[int] = []
        q = deque([start])
        visited.add(start)
        while q:
            fi = q.popleft()
            comp.append(fi)
            for e in face_edges[fi]:
                for nb in edge_to_faces[e]:
                    if nb not in visited:
                        visited.add(nb)
                        q.append(nb)
        islands.append(comp)

    islands.sort(key=len, reverse=True)
    return islands, no_uv


# ===========================================================================
# ===========================================================================

def _face_normals_and_areas(cache: _MeshCache, face_indices: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    fi = np.asarray(face_indices, dtype=np.int64)
    return cache.normals[fi], cache.areas[fi]


def _face_centroids(cache: _MeshCache, face_indices: Sequence[int]) -> np.ndarray:
    fi = np.asarray(face_indices, dtype=np.int64)
    return cache.centers[fi]


def _compute_island_snapshots(cache: _MeshCache,
                              islands: Sequence[Sequence[int]]) -> List[Tuple[int, float]]:
    out: List[Tuple[int, float]] = []
    for island in islands:
        fi = np.asarray(island, dtype=np.int64)
        if fi.size == 0:
            out.append((0, 0.0))
            continue
        out.append((int(fi.size), float(cache.areas[fi].sum())))
    return out


def _compute_island_normals(cache: _MeshCache,
                            islands: Sequence[Sequence[int]]) -> np.ndarray:
    out = np.zeros((len(islands), 3), dtype=np.float64)
    for ii, island in enumerate(islands):
        if not island:
            continue
        fi = np.asarray(island, dtype=np.int64)
        n = cache.normals[fi]
        a = cache.areas[fi]
        w = (n * a[:, None]).sum(axis=0)
        norm = float(np.linalg.norm(w))
        if norm > 1e-12:
            out[ii] = w / norm
    return out


def _build_island_adjacency_3d(mesh,
                               islands: Sequence[Sequence[int]],
                               canonical_v: Optional[np.ndarray] = None,
                               ) -> Dict[int, Set[int]]:
    face_to_island: Dict[int, int] = {}
    for ii, island in enumerate(islands):
        for fi in island:
            face_to_island[fi] = ii

    edge_to_islands: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
    for fi, (verts, _uvs, _mat) in enumerate(mesh.faces):
        if fi not in face_to_island:
            continue
        ii = face_to_island[fi]
        n = len(verts)
        for k in range(n):
            v1 = verts[k]
            v2 = verts[(k + 1) % n]
            if v1 == v2:
                continue
            if canonical_v is not None:
                v1 = int(canonical_v[v1])
                v2 = int(canonical_v[v2])
                if v1 == v2:
                    continue
            edge = (v1, v2) if v1 < v2 else (v2, v1)
            edge_to_islands[edge].add(ii)

    adj: Dict[int, Set[int]] = {ii: set() for ii in range(len(islands))}
    for islands_at_edge in edge_to_islands.values():
        if len(islands_at_edge) < 2:
            continue
        islands_list = list(islands_at_edge)
        for a in islands_list:
            for b in islands_list:
                if a != b:
                    adj[a].add(b)
    return adj


def _merge_small_islands(islands: Sequence[Sequence[int]],
                         snapshots: Sequence[Tuple[int, float]],
                         adjacency: Dict[int, Set[int]],
                         min_face_count: int,
                         min_area_ratio: float,
                         normals: Optional[np.ndarray] = None,
                         min_normal_cos: float = 0.5) -> List[List[int]]:
    n = len(islands)
    out: List[List[int]] = [list(faces) for faces in islands]

    if n == 0:
        return out

    total_area = sum(area for (_fc, area) in snapshots)
    if total_area <= 0.0:
        return out
    area_threshold = min_area_ratio * total_area

    is_small = [
        (fc < min_face_count) and (area < area_threshold)
        for (fc, area) in snapshots
    ]

    for x in range(n):
        if not is_small[x]:
            continue

        visited = {x}
        queue = deque([x])
        terminals: List[int] = []
        while queue:
            cur = queue.popleft()
            for nb in adjacency.get(cur, ()):
                if nb in visited:
                    continue
                visited.add(nb)
                if is_small[nb]:
                    queue.append(nb)
                else:
                    terminals.append(nb)

        if normals is not None and len(terminals) > 0:
            x_n = normals[x]
            if float(np.dot(x_n, x_n)) > 1e-12:
                terminals = [
                    t for t in terminals
                    if float(np.dot(x_n, normals[t])) >= min_normal_cos
                ]

        if not terminals:
            out[x] = []
            continue

        best = max(terminals, key=lambda j: snapshots[j][1])
        out[best].extend(out[x])
        out[x] = []

    return out


def _detect_seams_filtered(mesh,
                           face_to_island_merged: Dict[int, int],
                           canonical_v: Optional[np.ndarray] = None,
                           canonical_uv: Optional[np.ndarray] = None,
                           ) -> List[Tuple[int, int]]:
    edge_uv_map: Dict[Tuple[int, int], set] = defaultdict(set)
    edge_faces: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for fi, (verts, uvs, _mat) in enumerate(mesh.faces):
        if len(verts) != len(uvs):
            continue
        n = len(verts)
        for i in range(n):
            v1, v2 = verts[i], verts[(i + 1) % n]
            uv1, uv2 = uvs[i], uvs[(i + 1) % n]
            if canonical_v is not None:
                v1 = int(canonical_v[v1])
                v2 = int(canonical_v[v2])
            if canonical_uv is not None:
                uv1 = int(canonical_uv[uv1])
                uv2 = int(canonical_uv[uv2])
            edge = tuple(sorted((v1, v2)))
            edge_uv_map[edge].add(tuple(sorted((uv1, uv2))))
            edge_faces[edge].append(fi)

    seams: List[Tuple[int, int]] = []
    for edge, uvs_set in edge_uv_map.items():
        if len(uvs_set) < 2:
            continue
        islands = {face_to_island_merged.get(fi) for fi in edge_faces[edge]}
        if None in islands:
            continue
        if len(islands) < 2:
            continue
        seams.append(edge)
    return seams


def _seam_strips(seam_edges: Sequence[Tuple[int, int]]) -> List[List[int]]:
    adj: Dict[int, set] = defaultdict(set)
    for v1, v2 in seam_edges:
        adj[v1].add(v2)
        adj[v2].add(v1)
    visited: set = set()
    strips: List[List[int]] = []
    for v in adj:
        if v in visited:
            continue
        comp: List[int] = []
        stack = [v]
        visited.add(v)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in adj[cur]:
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        strips.append(comp)
    return strips


def _normal_coherence(cache: _MeshCache, faces: Sequence[int]) -> float:
    fi = np.asarray(faces, dtype=np.int64)
    if fi.size == 0:
        return 0.0
    nv = cache.normals[fi]
    av = cache.areas[fi]
    total = float(av.sum())
    if total <= 0.0:
        return 0.0
    return float(np.linalg.norm((nv * av[:, None]).sum(axis=0)) / total)


def _make_seam_views(mesh, cache: _MeshCache,
                     strips: Sequence[Sequence[int]],
                     fov_y_deg: float, aspect: float,
                     min_view_extent_ratio: float = 0.0,
                     split_normal_spread_deg: float = 0.0,
                     split_max_coherence: float = 0.5,
                     canonical_v: Optional[np.ndarray] = None,
                     ) -> List[ViewSpec]:
    if not strips:
        return []

    v_to_faces: Dict[int, List[int]] = defaultdict(list)
    for fi, (verts, _uvs, _mat) in enumerate(mesh.faces):
        for v in verts:
            key = int(canonical_v[v]) if canonical_v is not None else int(v)
            v_to_faces[key].append(fi)

    out: List[ViewSpec] = []
    for strip in strips:
        related: set = set()
        for v in strip:
            related.update(v_to_faces.get(v, ()))
        if not related:
            continue
        related_list = list(related)
        spread_trips = (
            split_normal_spread_deg > 0
            and _normal_spread_deg(cache, related_list) > split_normal_spread_deg
        )
        coherence_trips = (
            split_max_coherence > 0
            and _normal_coherence(cache, related_list) < split_max_coherence
        )
        if (spread_trips or coherence_trips) and len(related_list) >= 4:
            sub_groups = _split_island_by_normal(cache, related_list, 2)
        else:
            sub_groups = [related_list]
        for sub in sub_groups:
            v_obj = _make_island_view(
                mesh, cache, sub, fov_y_deg, aspect,
                min_view_extent_ratio=min_view_extent_ratio,
                source="seam",
            )
            if v_obj is not None:
                out.append(v_obj)
    return out


def select_views(
    mesh,
    *,
    fov_y_deg: float = 45.0,
    aspect: float = 1.0,
    split_curved_angle_deg: float = 50.0,
    min_view_extent_ratio: float = 0.30,
    merge_min_face_count: int = 200,
    merge_min_area_ratio: float = 0.02,
    seam_min_strip_vertices: int = 8,
    seam_split_normal_spread_deg: float = 0.0,
    seam_split_max_coherence: float = 0.5,
    weld_vertex_eps: float = 1e-6,
    weld_uv_eps: float = 1e-6,
    use_xatlas_charts: bool = False,
    log: bool = False,
) -> List[dict]:
    cache = _MeshCache.build(mesh)

    canonical_v = _compute_position_welding(mesh.vertices, weld_vertex_eps)
    canonical_uv: Optional[np.ndarray] = None
    is_per_vertex_uv = (
        hasattr(mesh, "uvs") and len(mesh.uvs) > 0
        and len(mesh.uvs) == len(mesh.vertices)
    )
    if is_per_vertex_uv and weld_uv_eps > 0:
        canonical_uv = _compute_position_welding(mesh.uvs, weld_uv_eps)
    if log:
        n_v = len(mesh.vertices)
        n_uv = len(getattr(mesh, "uvs", []))
        u_v = canonical_v.max() + 1 if canonical_v is not None else n_v
        if canonical_uv is not None:
            u_uv = canonical_uv.max() + 1
            uv_status = f"uvs {n_uv} -> {u_uv} canonical (eps={weld_uv_eps})"
        elif is_per_vertex_uv:
            uv_status = f"uvs welding disabled (eps={weld_uv_eps})"
        else:
            uv_status = (
                f"uvs {n_uv} (per-corner UV, welding skipped — chart-boundary "
                f"duplicates would be falsely merged)"
            )
        print(f"[view_selection] welding: verts {n_v} -> {u_v} canonical "
              f"(eps={weld_vertex_eps}); {uv_status}")

    faces_uvs_override: Optional[List[List[int]]] = None
    if use_xatlas_charts:
        faces_uvs_override = _compute_xatlas_uv_per_face(
            mesh, canonical_v=canonical_v, log=log,
        )
        if faces_uvs_override is not None:
            canonical_uv_for_islands = None
        else:
            canonical_uv_for_islands = canonical_uv
    else:
        canonical_uv_for_islands = canonical_uv

    islands, no_uv = _detect_uv_islands(
        mesh,
        canonical_v=canonical_v,
        canonical_uv=canonical_uv_for_islands,
        faces_uvs_override=faces_uvs_override,
    )
    if log:
        print(f"[view_selection] UV islands: {len(islands)} "
              f"(faces with no UV: {len(no_uv)})")

    if split_curved_angle_deg > 0:
        islands = _split_curved_islands(
            cache, islands, max_spread_deg=split_curved_angle_deg, log=log,
        )

    snapshots = _compute_island_snapshots(cache, islands)
    normals = _compute_island_normals(cache, islands)
    adjacency = _build_island_adjacency_3d(mesh, islands, canonical_v=canonical_v)
    island_merged = _merge_small_islands(
        islands, snapshots, adjacency,
        min_face_count=merge_min_face_count,
        min_area_ratio=merge_min_area_ratio,
        normals=normals,
    )
    if log:
        kept = sum(1 for f in island_merged if f)
        print(f"[view_selection] after merge: {kept} non-empty "
              f"(of {len(island_merged)})")

    face_to_island_merged: Dict[int, int] = {}
    for ii, faces in enumerate(island_merged):
        for fi in faces:
            face_to_island_merged[fi] = ii

    seam_edges = _detect_seams_filtered(
        mesh, face_to_island_merged,
        canonical_v=canonical_v, canonical_uv=canonical_uv,
    )
    strips = _seam_strips(seam_edges)
    if seam_min_strip_vertices > 0:
        strips = [s for s in strips if len(s) >= seam_min_strip_vertices]
    if log:
        print(f"[view_selection] seam edges (filtered): {len(seam_edges)}, "
              f"strips: {len(strips)} (>= {seam_min_strip_vertices} verts)")

    final: List[ViewSpec] = []
    for ii, faces in enumerate(island_merged):
        if not faces:
            continue
        v = _make_island_view(
            mesh, cache, faces, fov_y_deg, aspect,
            min_view_extent_ratio=min_view_extent_ratio,
            source="island",
        )
        if v is not None:
            final.append(v)

    final.extend(_make_seam_views(
        mesh, cache, strips, fov_y_deg, aspect,
        min_view_extent_ratio=min_view_extent_ratio,
        split_normal_spread_deg=seam_split_normal_spread_deg,
        split_max_coherence=seam_split_max_coherence,
        canonical_v=canonical_v,
    ))

    if len(final) < 6:
        final.extend(_make_cubemap_views(mesh, fov_y_deg, aspect))

    if log:
        srcs: dict = defaultdict(int)
        for v in final:
            srcs[v.source] += 1
        srcs_str = ", ".join(f"{k}={v}" for k, v in sorted(srcs.items()))
        print(f"[view_selection] FINAL: {len(final)} views ({srcs_str})")

    return [v.to_dict() for v in final]


# ===========================================================================
# ===========================================================================

def _make_cubemap_views(mesh, fov_y_deg: float = 45.0, aspect: float = 1.0) -> List[ViewSpec]:
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    center = (bbox_min + bbox_max) * 0.5
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    distance = diag * 1.5
    near = max(1e-3, distance - diag)
    far = distance + diag
    persp = {
        "fov_y_deg": float(fov_y_deg),
        "aspect": float(aspect),
        "near": float(near),
        "far": float(far),
    }
    directions = [
        (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0), (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0), (0.0, 0.0, -1.0),
    ]
    up = np.array([0.0, 1.0, 0.0])
    views = []
    for d in directions:
        eye = center + np.asarray(d) * distance
        view_mat = _lookat(eye, center, up)
        views.append(ViewSpec(
            view=view_mat.astype(np.float32),
            persp=persp,
            cam_pos=eye.astype(np.float32),
            forward=(center - eye).astype(np.float32),
            center=center.astype(np.float32),
            normal=(-np.asarray(d)).astype(np.float32),
            faces=[],
            source="cubemap",
        ))
    return views


# ===========================================================================
# ===========================================================================

def _normal_spread_deg(cache: _MeshCache, island: Sequence[int]) -> float:
    normals, areas = _face_normals_and_areas(cache, island)
    valid = areas > 0
    if valid.sum() < 2:
        return 0.0
    nv = normals[valid]
    av = areas[valid]
    w_n = (nv * av[:, None]).sum(axis=0)
    norm = float(np.linalg.norm(w_n))
    if norm < 1e-12:
        return 180.0
    mean_n = w_n / norm
    cos_min = float((nv @ mean_n).clip(-1.0, 1.0).min())
    return float(np.degrees(np.arccos(cos_min)))


def _split_island_by_normal(cache: _MeshCache, island: Sequence[int], k: int) -> List[List[int]]:
    normals, areas = _face_normals_and_areas(cache, island)
    valid = areas > 0
    if valid.sum() < k:
        return [list(island)]
    valid_idx = np.where(valid)[0]
    nv = normals[valid_idx]
    av = areas[valid_idx]

    order = np.argsort(-av)
    centers: List[np.ndarray] = []
    for li in order:
        if not centers:
            centers.append(nv[li])
            continue
        max_sim = max(float(np.dot(nv[li], c)) for c in centers)
        if (1.0 - max_sim) > 0.5:
            centers.append(nv[li])
        if len(centers) >= k:
            break
    if len(centers) < 2:
        return [list(island)]

    centers_arr = np.array(centers)
    for _ in range(20):
        sims = nv @ centers_arr.T
        labels = sims.argmax(axis=1)
        new_centers = []
        changed = False
        for c in range(centers_arr.shape[0]):
            mask = labels == c
            if not mask.any():
                new_centers.append(centers_arr[c])
                continue
            w_n = (nv[mask] * av[mask, None]).sum(axis=0)
            wn = float(np.linalg.norm(w_n))
            if wn < 1e-12:
                new_centers.append(centers_arr[c])
            else:
                nc = w_n / wn
                if float(np.dot(nc, centers_arr[c])) < 0.999:
                    changed = True
                new_centers.append(nc)
        centers_arr = np.array(new_centers)
        if not changed:
            break

    sims = nv @ centers_arr.T
    labels = sims.argmax(axis=1)
    out: List[List[int]] = []
    for c in range(centers_arr.shape[0]):
        members_local = np.where(labels == c)[0]
        if members_local.size == 0:
            continue
        out.append([int(island[valid_idx[i]]) for i in members_local])
    return out if len(out) > 1 else [list(island)]


def _split_curved_islands(cache: _MeshCache, islands: List[List[int]],
                          max_spread_deg: float = 50.0,
                          max_depth: int = 3,
                          log: bool = True) -> List[List[int]]:
    out: List[List[int]] = []
    queue: List[Tuple[List[int], int]] = [(isl, 0) for isl in islands]
    while queue:
        island, depth = queue.pop(0)
        spread = _normal_spread_deg(cache, island)
        if spread <= max_spread_deg or len(island) < 4 or depth >= max_depth:
            out.append(island)
            continue
        if spread < 90:
            k = 2
        elif spread < 135:
            k = 3
        else:
            k = 4
        sub = _split_island_by_normal(cache, island, k)
        if len(sub) <= 1:
            out.append(island)
            continue
        if log:
            print(f"[view_selection] split island ({len(island)} faces, "
                  f"spread={spread:.1f}°, depth={depth}) → {len(sub)} sub-islands")
        for s in sub:
            queue.append((s, depth + 1))
    return out


# ===========================================================================
# ===========================================================================

def _orthogonal_basis(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    candidates = (
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )
    abs_dots = [abs(float(np.dot(n, c))) for c in candidates]
    ref = candidates[int(np.argmin(abs_dots))]
    u = np.cross(ref, n)
    u /= max(1e-12, float(np.linalg.norm(u)))
    v = np.cross(n, u)
    v /= max(1e-12, float(np.linalg.norm(v)))
    return u, v


def _island_frame(mesh, cache: _MeshCache, island: Sequence[int]):
    normals, areas = _face_normals_and_areas(cache, island)
    cents = _face_centroids(cache, island)
    valid = areas > 0
    if valid.sum() == 0:
        return None
    nv = normals[valid]
    av = areas[valid]
    cv = cents[valid]
    total_area = float(av.sum())
    if total_area < 1e-14:
        return None

    center = (cv * av[:, None]).sum(axis=0) / total_area
    n_acc = (nv * av[:, None]).sum(axis=0)
    n_norm = float(np.linalg.norm(n_acc))
    if n_norm < 1e-9 * total_area:
        return None
    n = n_acc / n_norm

    fi_arr = np.asarray(island, dtype=np.int64)
    vert_idx = np.unique(cache.fv[fi_arr].ravel())
    if vert_idx.size == 0:
        return None
    verts = mesh.vertices[vert_idx]
    P = verts - center
    Pn = P @ n
    P_plane = P - np.outer(Pn, n)

    try:
        U, _, _ = np.linalg.svd(P_plane.T @ P_plane)
        u_raw = U[:, 0] - n * float(U[:, 0] @ n)
        un = float(np.linalg.norm(u_raw))
        u = u_raw / un if un > 1e-9 else _orthogonal_basis(n)[0]
    except np.linalg.LinAlgError:
        u = _orthogonal_basis(n)[0]
    v_axis = np.cross(n, u)
    v_axis /= max(1e-12, float(np.linalg.norm(v_axis)))

    su = P @ u
    sv = P @ v_axis
    sn = P @ n
    hx = 0.5 * float(su.max() - su.min())
    hy = 0.5 * float(sv.max() - sv.min())
    t = float(sn.max() - sn.min())
    return center, n, u, v_axis, hx, hy, t


# ===========================================================================
# ===========================================================================

def _build_camera(center: np.ndarray, n: np.ndarray,
                  fov_y_deg: float, aspect: float,
                  mesh, hx: float, hy: float, t: float,
                  margin_ratio: float = 0.05,
                  min_view_extent_ratio: float = 0.0,
                  participant_verts: Optional[np.ndarray] = None,
                  ) -> Optional[Tuple]:
    fov_y = math.radians(fov_y_deg)
    tan_y = math.tan(fov_y * 0.5)
    tan_x = tan_y * aspect
    margin = margin_ratio * max(hx, hy, 1e-6)

    dist_y = hy / tan_y if tan_y > 0 else 0.0
    dist_x = hx / tan_x if tan_x > 0 else 0.0
    dist_target = max(dist_x, dist_y) + 0.5 * t + margin

    if min_view_extent_ratio > 0:
        bbox_diag = float(np.linalg.norm(
            mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
        ))
        min_dist = (min_view_extent_ratio * bbox_diag) / (2.0 * tan_y)
        dist_target = max(dist_target, min_dist)

    cam_pos = center + n * dist_target

    forward = center - cam_pos
    fn = float(np.linalg.norm(forward))
    if fn < 1e-12:
        forward = -n
        fn = 1.0
    forward = forward / fn

    candidates = (
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )
    Y_DEGENERATE_COS = 0.99
    if abs(float(np.dot(forward, candidates[0]))) < Y_DEGENERATE_COS:
        world_up = candidates[0]
    else:
        abs_dots_xz = [
            abs(float(np.dot(forward, candidates[1]))),
            abs(float(np.dot(forward, candidates[2]))),
        ]
        world_up = candidates[1 + int(np.argmin(abs_dots_xz))]

    right = np.cross(forward, world_up)
    rn = float(np.linalg.norm(right))
    if rn < 1e-9:
        for c in candidates:
            right = np.cross(forward, c)
            rn = float(np.linalg.norm(right))
            if rn > 1e-9:
                break
    if rn < 1e-9:
        return None
    right /= rn
    up = np.cross(right, forward)
    up /= max(1e-12, float(np.linalg.norm(up)))

    rotation = np.eye(4, dtype=np.float64)
    rotation[0, :3] = right
    rotation[1, :3] = up
    rotation[2, :3] = -forward
    view = rotation.copy()
    view[:3, 3] = -rotation[:3, :3] @ cam_pos

    R = rotation[:3, :3]

    if participant_verts is not None and len(participant_verts) > 0:
        verts_part = np.asarray(mesh.vertices)[np.asarray(participant_verts)]
    else:
        verts_part = np.asarray(mesh.vertices)

    Q_pb = (verts_part - cam_pos) @ R.T
    z_pb = Q_pb[:, 2]
    if z_pb.size and z_pb.max() > 0:
        cam_pos = cam_pos + n * (float(z_pb.max()) + margin + 1e-4)
        view[:3, 3] = -rotation[:3, :3] @ cam_pos
        Q_pb = (verts_part - cam_pos) @ R.T
        z_pb = Q_pb[:, 2]

    z_pb_front = z_pb[z_pb < 0]
    if z_pb_front.size == 0:
        return None
    near = max(1e-3, float(-z_pb_front.max()) * 0.99)

    Q_all = (np.asarray(mesh.vertices) - cam_pos) @ R.T
    z_all = Q_all[:, 2]
    z_all_front = z_all[z_all < 0]
    if z_all_front.size == 0:
        return None
    far_candidate = float(-z_all_front.min()) * 1.01
    far_participant = float(-z_pb_front.min()) * 1.01
    far = max(near + 1e-3, far_participant, far_candidate)

    persp = {
        "fov_y_deg": float(fov_y_deg),
        "aspect": float(aspect),
        "near": float(near),
        "far": float(far),
    }
    return view, cam_pos, forward, persp


def _make_island_view(mesh, cache: _MeshCache, island: Sequence[int],
                      fov_y_deg: float, aspect: float,
                      min_view_extent_ratio: float = 0.0,
                      source: str = "island") -> Optional[ViewSpec]:
    frame = _island_frame(mesh, cache, island)
    if frame is None:
        return None
    center, n, _u, _v, hx, hy, t = frame
    if hx < 1e-9 and hy < 1e-9:
        return None

    fi_arr = np.asarray(island, dtype=np.int64)
    participant_verts = np.unique(cache.fv[fi_arr].ravel())

    primary = _build_camera(
        center, n, fov_y_deg, aspect, mesh, hx, hy, t,
        min_view_extent_ratio=min_view_extent_ratio,
        participant_verts=participant_verts,
    )
    if primary is None:
        return None
    p_view, p_cam, p_forward, p_persp = primary
    return ViewSpec(
        view=p_view, persp=p_persp,
        cam_pos=p_cam, forward=p_forward,
        center=center, normal=n,
        faces=list(island), source=source,
    )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def _lookat(eye, center, up, eps=1e-8):
    eye = np.asarray(eye, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    f = center - eye
    fn = np.linalg.norm(f)
    if fn < eps:
        raise ValueError("lookat: eye and center are too close or identical (forward vector ~ 0).")
    f = f / fn

    up_norm = up / (np.linalg.norm(up) + eps)
    if np.linalg.norm(up_norm) < eps:
        up_norm = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    if abs(np.dot(f, up_norm)) > 1.0 - 1e-6:
        alt = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(np.dot(f, alt)) > 1.0 - 1e-6:
            alt = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        up_norm = alt

    r = np.cross(f, up_norm)
    rn = np.linalg.norm(r)
    if rn < eps:
        raise RuntimeError("lookat: failed to compute a stable right vector (unexpected).")
    r = r / rn

    u = np.cross(r, f)

    M = np.eye(4, dtype=np.float32)
    M[0, 0:3] = r.astype(np.float32)
    M[1, 0:3] = u.astype(np.float32)
    M[2, 0:3] = (-f).astype(np.float32)
    M[0, 3] = -np.dot(r, eye).astype(np.float32)
    M[1, 3] = -np.dot(u, eye).astype(np.float32)
    M[2, 3] = np.dot(f, eye).astype(np.float32)

    if np.isnan(M).any():
        raise RuntimeError("lookat produced NaN in view matrix")

    return M
