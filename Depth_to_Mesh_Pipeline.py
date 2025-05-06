import torch
import numpy as np 
import os 
import cv2 
import open3d as o3d 
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage import measure 
import argparse
import glob
import skimage
import plyfile
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from DistDepth.utils import output_to_depth

from DistDepth.networks.resnet_encoder import ResnetEncoder
from DistDepth.networks.depth_decoder import DepthDecoder

class Depth_to_Mesh_Pipeline:
    def __init__(self, scannet_path, scene_id='scene_0241', subsample_factor=5, grid_resolution=256, marching_cubes_level=0.5, output_dir='output'):
        
        '''
        Initialize the pipeline for converting depth estimates to mesh
        
        Args:
            scannet_path: Path to the ScanNet dataset
            scene_id: Scene ID to process
            subsample_factor: Sample one frame for every subsample_factor frames
            grid_resolution: Resolution of the 3D grid for marching cubes
            marching_cubes_level: Isosurface level for marching cubes algorithm
            output_dir: Directory to save outputs
        '''

        self.scannet_path = scannet_path
        self.scene_id = scene_id
        self.scene_path = os.path.join(scannet_path, scene_id)
        self.subsample_factor = subsample_factor
        self.grid_resolution = grid_resolution
        self.marching_cubes_level = marching_cubes_level
        self.output_dir = output_dir

        #Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        #Load depth estimation model
        self.depth_model = self.load_distDepth_model()

        #Load camera parameters
        self.intrinsics, self.extrinsics = self._load_camera_parameters()


    #Loading DsitDepth Model
    def load_distDepth_model(self, device=None):
        '''
        Note device functionality is purely optional for the user to input. It is there to increase flexibility
        for the user. User can precisely control over which GPU/CPU the model lands on. If not defined we use 
        whatever pytorch thinks is the best.
        '''
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt_dir = "/Users/muneebaliasif/Documents/EIC/ckpts"
        
        #Building the encoder and loading its weights
        encoder = ResnetEncoder(152, False)
        loaded_enc = torch.load(os.path.join(ckpt_dir, "encoder.pth"), map_location=device)

        #Filtering out any unexpected keys
        filtered_enc = {
        k: v for k, v in loaded_enc.items()
        if k in encoder.state_dict()
        }
        encoder.load_state_dict(filtered_enc)
        encoder.to(device).eval()

        #Building the decoder and loading its weights
        depth_decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
        loaded_dec = torch.load(os.path.join(ckpt_dir, "depth.pth"), map_location=device)
        depth_decoder.load_state_dict(loaded_dec)
        depth_decoder.to(device).eval()

        return encoder, depth_decoder
    
    def _load_camera_parameters(self):
        '''
        Load camera intrinsics and extrinsics from ScanNet
        '''

        # Load intrinsics
        intrinsic_file = os.path.join(self.scene_path, f"{self.scene_id}.txt")
        with open(intrinsic_file, 'r') as f:
            intrinsic_data = f.readlines()
        
        # Parse intrinsic matrix
        intrinsic = np.eye(3)
        for i in range(3):
            line = intrinsic_data[i].strip().split()
            intrinsic[i, 0] = float(line[0])
            intrinsic[i, 1] = float(line[1])
            intrinsic[i, 2] = float(line[2])
        
        # Load extrinsics (poses)
        pose_files = sorted(glob.glob(os.path.join(self.scene_path, 'pose', '*.txt')))
        extrinsics = []
        
        for pose_file in pose_files:
            with open(pose_file, 'r') as f:
                pose_data = f.readlines()
            pose = np.eye(4)
            for i in range(4):
                line = pose_data[i].strip().split()
                pose[i, 0] = float(line[0])
                pose[i, 1] = float(line[1])
                pose[i, 2] = float(line[2])
                pose[i, 3] = float(line[3])
            extrinsics.append(pose)
        
        return intrinsic, extrinsics
    
    def preprocess_image(self, image_path):
        '''
        Preprocess the image for depth estimation
        '''
    
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to 640x480
        image = cv2.resize(image, (640, 480))
        
        # Center crop to 624x468 (to avoid camera distortion at edges)
        h, w = image.shape[:2]
        crop_h, crop_w = 468, 624
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        cropped_image = image[start_h:start_h+crop_h, start_w:start_w+crop_w]
        
        # Convert to torch tensor and normalize
        input_tensor = torch.from_numpy(cropped_image.transpose((2, 0, 1))).float() / 255.0
        
        #Add batch dimension
        input_tensor = input_tensor.unsqueeze(0)

        #Resizing to networks expected 256x256
        input_tensor = F.interpolate(input_tensor, size=(256, 256), mode="bilinear", align_corners=False)
        
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
            
        return input_tensor, cropped_image
    
    def estimate_depth(self, image_path):
        '''
        Estimate depth from an RGB image using DistDepth
        '''
        input_tensor, original_image = self.preprocess_image(image_path)
        device = next(self.depth_model[0].parameters()).device  # Get device from model
        
        if input_tensor.device != device:
            input_tensor = input_tensor.to(device)
        
        with torch.no_grad():
            # Unpack encoder and decoder from the model tuple
            encoder, decoder = self.depth_model
            
            # Get encoder features
            features = encoder(input_tensor)
            
            # Decode features to get depth
            outputs = decoder(features)
            
            # Get depth from the output
            #depth = outputs[("out", 0)]
            disp = outputs[("out", 0)]
            depth = output_to_depth(disp, 0.1, 10)
            
        # Convert depth tensor to numpy array
        depth_np = depth.cpu().numpy()[0, 0]
        
        return depth_np, original_image
    
    def depth_to_point_cloud(self, depth_map, extrinsic, index=0):
        '''
        Convert depth map to point cloud
        
        Args:
            depth_map: Depth map from model
            extrinsic: Camera extrinsic matrix (camera pose)
            index: Index of the frame (for saving the first frame separately)
        
        Returns:
            point_cloud: Open3D point cloud
        '''
        h, w = depth_map.shape
        
        # Create mesh grid for pixel coordinates
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        # Adjust intrinsic for cropped image
        fx = self.intrinsics[0, 0]
        fy = self.intrinsics[1, 1]
        cx = self.intrinsics[0, 2] - 8  # Adjust for cropping (640-624)/2 = 8
        cy = self.intrinsics[1, 2] - 6  # Adjust for cropping (480-468)/2 = 6
        
        # Convert pixel coordinates to 3D points in camera coordinate system
        x_cam = (x - cx) * depth_map / fx
        y_cam = (y - cy) * depth_map / fy
        z_cam = depth_map
        
        # Flatten and filter out invalid depth points
        x_cam = x_cam.flatten()
        y_cam = y_cam.flatten()
        z_cam = z_cam.flatten()
        
        valid_mask = z_cam > 0
        x_cam = x_cam[valid_mask]
        y_cam = y_cam[valid_mask]
        z_cam = z_cam[valid_mask]
        
        # Stack points
        points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
        
        # Create homogeneous coordinates
        points_hom = np.hstack([points_cam, np.ones((points_cam.shape[0], 1))])
        
        # Transform points to world coordinate system
        points_world_hom = np.dot(extrinsic, points_hom.T).T
        points_world = points_world_hom[:, :3]
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_world)
        
        # If this is the first frame, save the point cloud
        if index == 0:
            self._save_point_cloud(pcd, os.path.join(self.output_dir, 'first_frame_point_cloud.ply'))
            
        return pcd, points_world
    
    def create_mesh_from_point_cloud(self, all_points):
        """
        Create mesh from point cloud using 3D grid and marching cubes
        
        Args:
            all_points: List of points from all frames
        
        Returns:
            mesh_path: Path to the saved mesh file
        """
        # # Concatenate all points
        # all_points = np.vstack(all_points)
        
        # # Get min and max bounds
        # min_bound = np.min(all_points, axis=0)
        # max_bound = np.max(all_points, axis=0)
        
        # # Create bounding box tensor
        # bbox = np.stack([min_bound, max_bound], axis=0)
        
        # # Create 3D grid
        # grid_size = self.grid_resolution
        # grid = np.zeros((grid_size, grid_size, grid_size))
        
        # # Calculate voxel size
        # voxel_size = (max_bound - min_bound) / grid_size
        
        # # Populate grid
        # for point in tqdm(all_points, desc="Creating 3D grid"):
        #     # Convert world coordinates to grid indices
        #     idx = np.floor((point - min_bound) / voxel_size).astype(int)
            
        #     # Ensure indices are within bounds
        #     if np.all(idx >= 0) and np.all(idx < grid_size):
        #         grid[idx[0], idx[1], idx[2]] += 1
        
        # # Normalize grid values to [0, 1]
        # if np.max(grid) > 0:
        #     grid = grid / np.max(grid)
        
        # # Apply Gaussian filter for smoothing
        # from scipy.ndimage import gaussian_filter
        # grid = gaussian_filter(grid, sigma=1.0)
        
        # # Invert the grid to make it an SDF (high values should be empty space)
        # grid = 1.0 - grid
        
        # # Convert numpy grid to torch tensor
        # grid_tensor = torch.from_numpy(grid).float()
        
        # # Define mesh output path
        # mesh_path = os.path.join(self.output_dir, 'reconstructed_mesh.ply')
        
        # # Convert grid to mesh using the provided function
        # self.convert_sdf_samples_to_ply(
        #     grid_tensor,
        #     mesh_path,
        #     bbox=torch.from_numpy(bbox).float(),
        #     level=self.marching_cubes_level
        # )
        
        # return mesh_path

        pts = np.vstack(all_points)

        # 2) build bbox as NumPy array (keep as np.ndarray!)
        min_bound = pts.min(axis=0)
        max_bound = pts.max(axis=0)
        bbox = np.stack([min_bound, max_bound], axis=0)   # shape (2,3)

        # 3) build SDF volume
        sdf = self._compute_sdf_volume(pts, bbox, self.grid_resolution)
        # sdf is np.ndarray (res, res, res)

        # 4) turn into a torch tensor (optional) or just pass ndarray
        sdf_tensor = torch.from_numpy(sdf)

        # 5) choose level=0 and call your converter
        mesh_path = os.path.join(self.output_dir, 'reconstructed_mesh.ply')
        self.convert_sdf_samples_to_ply(
            sdf_tensor,
            mesh_path,
            bbox=bbox,
            level=0.0
        )

        return mesh_path
    
     
    def _compute_sdf_volume(self, points: np.ndarray, bbox: np.ndarray, res: int):
        """
        points: (Px3) world-space points
        bbox:   (2x3) array [[min_x,min_y,min_z],[max_x,max_y,max_z]]
        res:    grid resolution (N → NxNxN)
        Returns a (NxNxN) signed-distance field (float32).
        """
        # 1) build empty occupancy grid
        grid = np.zeros((res, res, res), dtype=bool)

        # 2) rasterize points → voxels
        voxel_size = (bbox[1] - bbox[0]) / (res - 1)
        idx = ((points - bbox[0]) / voxel_size).astype(int)
        valid = np.all((idx >= 0) & (idx < res), axis=1)
        ix, iy, iz = idx[valid].T
        grid[ix, iy, iz] = True

        # 3) distance transforms (outside vs inside)
        outside = distance_transform_edt(~grid) * voxel_size[0]
        inside  = distance_transform_edt(grid)  * voxel_size[0]

        # 4) signed‐distance: positive outside, negative inside
        sdf = (outside - inside).astype(np.float32)
        return sdf
    

    def _save_point_cloud(self, point_cloud, file_path):
        """Save point cloud to file"""
        o3d.io.write_point_cloud(file_path, point_cloud)
        print(f"Point cloud saved to {file_path}")
    
    
    def convert_sdf_samples_to_ply(self, pytorch_3d_sdf_tensor: torch.Tensor, ply_filename_out: str, bbox: np.ndarray, level: float = 0.3, offset: np.ndarray = None, scale: float = None):
        """
        Convert SDF samples to a PLY mesh using Open3D for correct PLY formatting.

        Args:
            pytorch_3d_sdf_tensor: torch.FloatTensor of shape (N,N,N)
            ply_filename_out: Path to save the output PLY
            bbox: numpy array shape (2,3) with [min_bound, max_bound]
            level: isosurface level for marching cubes
            offset: optional offset to subtract from vertices
            scale: optional scale to divide vertices
        Returns:
            Absolute path to the saved PLY file.
        """
        # 1) Convert to numpy volume and compute voxel spacing
        vol = pytorch_3d_sdf_tensor.numpy()
        # (max-min)/(grid_size-1) gives real-world spacing
        voxel_size = (bbox[1] - bbox[0]) / (np.array(vol.shape) - 1)

        # 2) Run marching cubes to extract the mesh
        verts, faces, normals, _ = skimage.measure.marching_cubes(
            vol, level=level, spacing=voxel_size
        )
        # Invert face orientation for Open3D
        faces = faces[:, ::-1]

        # 3) Map vertex positions from voxel to world coordinates
        mesh_pts = verts + bbox[0]  # bbox[0] is the min corner

        # 4) Apply optional scale/offset
        if scale is not None:
            mesh_pts = mesh_pts / scale
        if offset is not None:
            mesh_pts = mesh_pts - offset

        # 5) Build an Open3D TriangleMesh
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(mesh_pts),
            o3d.utility.Vector3iVector(faces)
        )
        mesh.compute_vertex_normals()

        # 6) Write out a valid ASCII PLY
        print(f"Saving mesh to {ply_filename_out}")
        o3d.io.write_triangle_mesh(ply_filename_out, mesh, write_ascii=True)

        return os.path.abspath(ply_filename_out)

    
    def evaluate_mesh_quality(self, reconstructed_mesh_path, gt_mesh_path):
        """
        Evaluate mesh quality compared to ground truth
        
        Args:
            reconstructed_mesh_path: Path to reconstructed mesh
            gt_mesh_path: Path to ground truth mesh
        
        Returns:
            metrics: Dictionary of quality metrics
        """
        #def evaluate_mesh_quality(self, reconstructed_mesh_path, gt_mesh_path):
        # Load meshes
        reconstructed_mesh = o3d.io.read_triangle_mesh(reconstructed_mesh_path)
        gt_mesh = o3d.io.read_triangle_mesh(gt_mesh_path)
        
        # Compute metrics
        # For simplicity, we'll use Chamfer distance as a metric
        # Convert meshes to point clouds for distance computation
        reconstructed_pcd = reconstructed_mesh.sample_points_uniformly(number_of_points=100000)
        gt_pcd = gt_mesh.sample_points_uniformly(number_of_points=100000)
        
        # Compute distances
        distances1 = np.asarray(reconstructed_pcd.compute_point_cloud_distance(gt_pcd))
        distances2 = np.asarray(gt_pcd.compute_point_cloud_distance(reconstructed_pcd))
        
        chamfer_distance = np.mean(distances1) + np.mean(distances2)
        
        metrics = {
            'chamfer_distance': chamfer_distance,
            'reconstructed_vertices': len(reconstructed_mesh.vertices),
            'reconstructed_triangles': len(reconstructed_mesh.triangles),
            'gt_vertices': len(gt_mesh.vertices),
            'gt_triangles': len(gt_mesh.triangles)
        }
        
        return metrics
        
    
    def run_pipeline(self):
        """Run the complete pipeline"""
        # Get list of RGB images
        image_files = sorted(glob.glob(os.path.join(self.scene_path, 'color', '*.jpg')))
        
        # Subsample frames
        image_files = image_files[::self.subsample_factor]
        
        all_points = []
        
        # Process each image
        for i, image_file in enumerate(tqdm(image_files, desc="Processing images")):
            # Extract frame index
            frame_idx = int(os.path.basename(image_file).split('.')[0])
            
            # Estimate depth
            depth_map, _ = self.estimate_depth(image_file)
            
            # Convert depth to point cloud
            _, points_world = self.depth_to_point_cloud(
                depth_map, 
                self.extrinsics[frame_idx], 
                index=i
            )
            
            all_points.append(points_world)
        
        # Create mesh from all point clouds
        mesh_path = self.create_mesh_from_point_cloud(all_points)
                
        return mesh_path

def main():
    parser = argparse.ArgumentParser(description="Depth to Mesh Pipeline")
    parser.add_argument("--scannet_path", required=True, help="Path to ScanNet dataset")
    parser.add_argument("--scene_id", default="scene_0241", help="Scene ID to process")
    parser.add_argument("--subsample", type=int, default=5, help="Subsample factor")
    parser.add_argument("--grid_resolution", type=int, default=256, help="3D grid resolution")
    parser.add_argument("--mc_level", type=float, default=0.5, help="Marching cubes level")
    parser.add_argument("--output_dir", default="output", help="Output directory")
    
    args = parser.parse_args()
    
    pipeline = Depth_to_Mesh_Pipeline(
        scannet_path=args.scannet_path,
        scene_id=args.scene_id,
        subsample_factor=args.subsample,
        grid_resolution=args.grid_resolution,
        marching_cubes_level=args.mc_level,
        output_dir=args.output_dir
    )
    
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()
        


