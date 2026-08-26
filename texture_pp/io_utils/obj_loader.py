"""OBJ mesh loader."""

import numpy as np

class OBJ:

    def __init__(self, obj_file_path: str):
        self.obj_file_path = obj_file_path
        self.parse()
        self.scale_mesh_to_unit_sphere()
        self.mesh_info = self.analyze_mesh()

    def parse(self):
        vertices = []
        uvs = []
        normals = []

        self.faces = []

        with open(self.obj_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                if len(parts) == 0:
                    continue

                if parts[0] == 'v':
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

                elif parts[0] == 'vt':
                    uvs.append([float(parts[1]), float(parts[2])])

                elif parts[0] == 'vn':
                    normals.append([float(parts[1]), float(parts[2]), float(parts[3])])

                elif parts[0] == 'f':
                    face_verts = []
                    face_uvs = []
                    face_normals = []

                    for vertex in parts[1:]:
                        indices = vertex.split('/')
                        if len(indices) >= 1 and indices[0]:
                            face_verts.append(int(indices[0]) - 1)
                            if len(indices) >= 2 and indices[1]:
                                face_uvs.append(int(indices[1]) - 1)
                            else:
                                face_uvs.append(0)
                            if len(indices) >= 3 and indices[2]:
                                face_normals.append(int(indices[2]) - 1)
                            else:
                                face_normals.append(0)

                    if len(face_verts) >= 3:
                        for k in range(1, len(face_verts) - 1):
                            self.faces.append((
                                [face_verts[0], face_verts[k], face_verts[k + 1]],
                                [face_uvs[0], face_uvs[k], face_uvs[k + 1]],
                                [face_normals[0], face_normals[k], face_normals[k + 1]],
                            ))

        self.vertices = np.array(vertices) if vertices else np.array([])
        self.vert_idx = np.array([face[0] for face in self.faces]) if self.faces else np.array([])
        self.uvs = np.array(uvs) if uvs else np.array([])
        self.uv_idx = np.array([face[1] for face in self.faces]) if self.faces else np.array([])
        self.normal_idx = np.array([face[2] for face in self.faces]) if self.faces else np.array([])
        self.normals = np.array(normals) if normals else np.array([])

        # meshes without vn data: derive per-face normals from the geometry
        if len(self.normals) == 0 and len(self.faces) > 0:
            fv = np.array([f[0] for f in self.faces], dtype=np.int64)
            v0 = self.vertices[fv[:, 0]]
            v1 = self.vertices[fv[:, 1]]
            v2 = self.vertices[fv[:, 2]]
            n = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(n, axis=1, keepdims=True)
            n = n / np.where(norm > 1e-12, norm, 1.0)
            self.normals = n.astype(np.float32)
            self.normal_idx = np.repeat(np.arange(len(self.faces), dtype=np.int64)[:, None], 3, axis=1)

        print(f"Mesh info: {len(self.vertices)} vertexs, {len(self.uvs)} uv_coords, {len(self.faces)} faces")

        if not self.faces:
            self.unwelded_vertices = np.empty((0, 3), dtype=np.float32)
            self.unwelded_uvs = np.empty((0, 2), dtype=np.float32)
            self.unwelded_normals = np.empty((0, 3), dtype=np.float32)
            self.unwelded_indices = np.empty((0, 3), dtype=np.int32)
            print("Mesh has no faces, skipping unweld.")
            return

        face_vert_indices = np.array([f[0] for f in self.faces], dtype=np.int64)
        face_uv_indices = np.array([f[1] for f in self.faces], dtype=np.int64)
        face_normal_indices = np.array([f[2] for f in self.faces], dtype=np.int64)

        self.unwelded_vertices = self.vertices[face_vert_indices.flatten()]
        self.unwelded_uvs = self.uvs[face_uv_indices.flatten()]
        self.unwelded_normals = self.normals[face_normal_indices.flatten()]

        num_corners = len(self.unwelded_vertices)
        self.unwelded_indices = np.arange(num_corners, dtype=np.int32).reshape(-1, 3)

    def scale_mesh_to_unit_sphere(self):
        bbox_min = np.min(self.vertices, axis=0)
        bbox_max = np.max(self.vertices, axis=0)
        center = (bbox_min + bbox_max) / 2.0
        extent = bbox_max - bbox_min
        max_extent = np.max(extent)
        self.vertices -= center
        scale_factor = 1.0 / (max_extent / 2.0)
        self.vertices *= scale_factor
        self.recompute_unwelded()

    def recompute_unwelded(self):
        if not hasattr(self, 'faces') or len(self.faces) == 0:
            self.unwelded_vertices = np.empty((0, 3), dtype=np.float32)
            self.unwelded_uvs = np.empty((0, 2), dtype=np.float32)
            self.unwelded_normals = np.empty((0, 3), dtype=np.float32)
            self.unwelded_indices = np.empty((0, 3), dtype=np.int32)
            return
        face_vert_indices = np.array([f[0] for f in self.faces], dtype=np.int64)
        face_uv_indices = np.array([f[1] for f in self.faces], dtype=np.int64)
        face_normal_indices = np.array([f[2] for f in self.faces], dtype=np.int64)

        self.unwelded_vertices = self.vertices[face_vert_indices.flatten()].astype(np.float32)
        self.unwelded_uvs = self.uvs[face_uv_indices.flatten()].astype(np.float32) if len(self.uvs) else np.empty((0,2), dtype=np.float32)
        self.unwelded_normals = self.normals[face_normal_indices.flatten()].astype(np.float32) if len(self.normals) else np.empty((0,3), dtype=np.float32)
        num_corners = len(self.unwelded_vertices)
        self.unwelded_indices = np.arange(num_corners, dtype=np.int32).reshape(-1, 3)

    def analyze_mesh(self):
        bbox_min = np.min(self.vertices, axis=0)
        bbox_max = np.max(self.vertices, axis=0)
        center = (bbox_min + bbox_max) / 2.0
        extent = bbox_max - bbox_min
        size = np.linalg.norm(extent)
        max_extent = np.max(extent)

        fov = 90.0
        fov_radians = np.radians(fov)
        camera_distance = 3.0
        near_plane = 0.1
        far_plane = 4

        return {
            'bbox_min': bbox_min,
            'bbox_max': bbox_max,
            'center': center,
            'size': size,
            'extent': extent,
            'max_extent': max_extent,
            'camera_distance': camera_distance,
            'near_plane': near_plane,
            'far_plane': far_plane,
            'fov': fov
        }
