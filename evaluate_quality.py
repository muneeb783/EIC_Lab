# evaluate_quality.py
import os
import open3d as o3d
from Depth_to_Mesh_Pipeline import Depth_to_Mesh_Pipeline

# Instantiate your pipeline (parameters don’t actually matter here)
pipe = Depth_to_Mesh_Pipeline(
    scannet_path="scannet",
    scene_id="scene_0241",
    subsample_factor=5,
    grid_resolution=256,
    marching_cubes_level=0.0,
    output_dir="output"
)

# Paths to the meshes you already generated
recon = os.path.join("output", "reconstructed_mesh.ply")
gt    = os.path.join("scannet_raw", "scans", "scene0241_01", "scene0241_01_vh_clean_2.ply")

# Sanity check - to confirm that a valid file was downloaded
gt_mesh = o3d.io.read_triangle_mesh(gt)
print("GT mesh:", gt)
print("  # vertices:", len(gt_mesh.vertices))
print("  # triangles:", len(gt_mesh.triangles))

# Run the evaluation
metrics = pipe.evaluate_mesh_quality(recon, gt)
print("=== Mesh Quality Metrics ===")
for k, v in metrics.items():
    print(f"{k:25s}: {v:.6f}")
