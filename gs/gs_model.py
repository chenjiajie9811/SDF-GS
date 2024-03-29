import os
import torch
import numpy as np

from torch import nn

from plyfile import PlyData, PlyElement

from simple_knn._C import distCUDA2

from utils.mesh import Mesh
from utils.mesh_utils import decimate_mesh, clean_mesh, sample_points_with_barycentric
from utils.system_utils import mkdir_p
from utils.sh_utils import eval_sh, SH2RGB, RGB2SH
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from utils.rigid_utils import matrix_to_quaternion

from grid.grid import DenseGrid, Grid2Mesh
from grid.network import Mesh2GaussiansNetwork, GaussianNetwork, DepthGaussianNetwork

from skimage import measure
from scipy.spatial.transform import Rotation as R

def gaussian_3d_coeff(xyzs, covs):
    # xyzs: [N, 3] : query_xyzs - gaussian_center_xyzs
    # covs: [N, 6]
    x, y, z = xyzs[:, 0], xyzs[:, 1], xyzs[:, 2]
    a, b, c, d, e, f = covs[:, 0], covs[:, 1], covs[:, 2], covs[:, 3], covs[:, 4], covs[:, 5]

    # eps must be small enough !!!
    inv_det = 1 / (a * d * f + 2 * e * c * b - e**2 * a - c**2 * d - b**2 * f + 1e-24)
    inv_a = (d * f - e**2) * inv_det
    inv_b = (e * c - b * f) * inv_det
    inv_c = (e * b - c * d) * inv_det
    inv_d = (a * f - c**2) * inv_det
    inv_e = (b * c - e * a) * inv_det
    inv_f = (a * d - b**2) * inv_det

    power = -0.5 * (x**2 * inv_a + y**2 * inv_d + z**2 * inv_f) - x * y * inv_b - x * z * inv_c - y * z * inv_e

    power[power > 0] = -1e10 # abnormal values... make weights 0
        
    return torch.exp(power)


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, device='cuda'):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.device = device
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @torch.no_grad()
    def extract_fields(self, resolution=128, num_blocks=16, relax_ratio=1.5):
        # resolution: resolution of field
        self.resolution = [resolution, resolution, resolution]

        print("Extracting density fields...")
        
        block_size = 2 / num_blocks # 2 : -1 ~ 1

        assert resolution % block_size == 0
        split_size = resolution // num_blocks

        opacities = self.get_opacity

        # pre-filter low opacity gaussians to save computation
        mask = (opacities > 0.005).squeeze(1)

        opacities = opacities[mask]
        xyzs = self.get_xyz[mask]
        stds = self.get_scaling[mask]
        
        # normalize to ~ [-1, 1]
        mn, mx = xyzs.amin(0), xyzs.amax(0)
        self.grid_center = (mn + mx) / 2
        self.grid_scale = 1.8 / (mx - mn).amax().item() 
        self.xyz_min = mn
        self.xyz_max = mx

        xyzs = (xyzs - self.grid_center) * self.grid_scale
        stds = stds * self.grid_scale

        covs = self.covariance_activation(stds, 1, self._rotation[mask]) # symmetric matrix with 6 parameters for each gaussian

        # tile
        device = opacities.device
        occ = torch.zeros([resolution] * 3, dtype=torch.float32, device=device)

        X = torch.linspace(-1, 1, resolution).split(split_size)
        Y = torch.linspace(-1, 1, resolution).split(split_size)
        Z = torch.linspace(-1, 1, resolution).split(split_size)

        # loop blocks (assume max size of gaussian is small than relax_ratio * block_size !!!)
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    # sample fine grid points inside each coarse voxel
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    # sample points [M, 3] M : number of query points inside this coarse voxel
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).to(device)
                    # in-tile gaussians mask
                    vmin, vmax = pts.amin(0), pts.amax(0)
                    vmin -= block_size * relax_ratio
                    vmax += block_size * relax_ratio
                    mask = (xyzs < vmax).all(-1) & (xyzs > vmin).all(-1)
                    # if hit no gaussian, continue to next block
                    if not mask.any():
                        continue
                    # L : number of Gaussians inside this coarse voxel
                    mask_xyzs = xyzs[mask] # [L, 3]
                    mask_covs = covs[mask] # [L, 6]
                    mask_opas = opacities[mask].view(1, -1) # [L, 1] --> [1, L]

                    # query per point-gaussian pair.
                    # The distance between each query point (M) and each gaussian center (L)
                    g_pts = pts.unsqueeze(1).repeat(1, mask_covs.shape[0], 1) - mask_xyzs.unsqueeze(0) # [M, L, 3]
                    g_covs = mask_covs.unsqueeze(0).repeat(pts.shape[0], 1, 1) # [M, L, 6]

                    # batch on gaussian to avoid OOM
                    batch_g = 1024
                    val = 0
                    for start in range(0, g_covs.shape[1], batch_g):
                        end = min(start + batch_g, g_covs.shape[1])
                        w = gaussian_3d_coeff(g_pts[:, start:end].reshape(-1, 3), g_covs[:, start:end].reshape(-1, 6)).reshape(pts.shape[0], -1) # [M, l]
                        val += (mask_opas[:, start:end] * w).sum(-1)
                    
                    occ[xi * split_size: xi * split_size + len(xs), 
                        yi * split_size: yi * split_size + len(ys), 
                        zi * split_size: zi * split_size + len(zs)] = val.reshape(len(xs), len(ys), len(zs)) 
    
        return occ
    
    def extract_mesh(self, path, density_thresh=1, resolution=128, decimate_target=1e5):

        os.makedirs(os.path.dirname(path), exist_ok=True)

        occ = self.extract_fields(resolution).detach().cpu().numpy()

        print ("Running marching cubes...")
        # import mcubes
        # vertices, triangles = mcubes.marching_cubes(occ, density_thresh)

        from skimage import measure
        vertices, triangles, normals, values = measure.marching_cubes(occ, level=density_thresh)

        # occ = self.extract_fields(resolution)#.detach().cpu().numpy()

        # print ("Running marching cubes...")
        # import cumcubes
        # vertices, triangles = cumcubes.marching_cubes(occ, density_thresh)
        # vertices = vertices.detach().cpu().numpy()
        # triangles = triangles.detach().cpu().numpy()

        vertices = vertices / (resolution - 1.0) * 2 - 1

        # transform back to the original space
        vertices = vertices / self.grid_scale + self.grid_center.detach().cpu().numpy()

        vertices, triangles = clean_mesh(vertices, triangles, remesh=True, remesh_size=0.015)
        if decimate_target > 0 and triangles.shape[0] > decimate_target:
            vertices, triangles = decimate_mesh(vertices, triangles, decimate_target)

        v = torch.from_numpy(vertices.astype(np.float32)).contiguous().to(self.device).double()
        f = torch.from_numpy(triangles.astype(np.int32)).contiguous().to(self.device).long()

        print(
            f"[INFO] marching cubes result: {v.shape} ({v.min().item()}-{v.max().item()}), {f.shape}"
        )

        mesh = Mesh(v=v, f=f, device='cuda')

        return mesh
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float = 1):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().to(self.device)
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().to(self.device))
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().to(self.device)
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device=self.device)
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device=self.device))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.device)

    def create_from_mesh(self, mesh : Mesh, spatial_lr_scale : float = 1):
        if mesh.vn is None:
            mesh.auto_normal()
        print("Reinitializing from mesh...")
        
        self.spatial_lr_scale = spatial_lr_scale
        sampled_points = sample_points_with_barycentric(mesh)
        shs = np.random.random((sampled_points.shape[0], 3)) / 255.0
        
        pcd = BasicPointCloud(points=sampled_points, colors=SH2RGB(shs), normals=np.zeros((sampled_points.shape[0], 3)))
        self.create_from_pcd(pcd)



    def training_setup(self, training_args):
        self.percent_dense = training_args['percent_dense']
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)

        l = [
            {'params': [self._xyz], 'lr': training_args['position_lr_init'] * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args['feature_lr'], "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args['feature_lr'] / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args['opacity_lr'], "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args['scaling_lr'], "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args['rotation_lr'], "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args['position_lr_init']*self.spatial_lr_scale,
                                                    lr_final=training_args['position_lr_final']*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args['position_lr_delay_mult'],
                                                    max_steps=training_args['position_lr_max_steps'])
        
        

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
            if param_group["name"] == 'sdf_net':
                lr = self.sdf_net_lr_scheduler(iteration)
                param_group['lr'] = lr
            if param_group["name"] == 'gauss_geo_net':
                lr = self.gauss_geo_net_lr_scheduler(iteration)
                param_group['lr'] = lr
            if param_group["name"] == 'gauss_rgb_net':
                lr = self.gauss_rgb_net_lr_scheduler(iteration)
                param_group['lr'] = lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save_simple_ply_color(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

        colors = SH2RGB(f_dc) * 255
        vertex = np.array([(xyz[i][0], xyz[i][1], xyz[i][2], colors[i][0], colors[i][1], colors[i][2]) 
                           for i in range(xyz.shape[0])], 
                           dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                         ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        
        vertex_element = PlyElement.describe(vertex, 'vertex')
        plydata = PlyData([vertex_element])
        plydata.write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        print("Number of points at loading : ", xyz.shape[0])

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device=self.device).requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device=self.device).transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device=self.device).transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device=self.device).requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device=self.device).requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device=self.device).requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.device)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.device)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device=self.device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device=self.device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device=self.device, dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent
        )
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def prune(self, min_opacity, extent, max_screen_size):

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()


    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1


    # Joint optimization after extracting coarse mesh
    def generate_bary_coords_params(self, num_faces):
        # [K, 3] K : number of sampled points on a triangle
        default_barycentric_weights = torch.tensor([
                                                    [1./3, 1./3, 1./3], 
                                                    [1./2, 1./4, 1./4], 
                                                    [1./4, 1./2, 1./4],
                                                    [1./4, 1./4, 1./2]])
        

        new_centers_bary_weights = default_barycentric_weights.unsqueeze(0).repeat(num_faces, 1, 1) # [N, K, 3]
        
        # print (new_centers_bary_weights.shape)
        return new_centers_bary_weights
    

    def generate_rotations_from_triangles(self, v, f, normals):
        # Last column of the rotation matrix is the triangle surface normal
        # one column is an edge of the triangle
        # the other one is compyted by cross product
        rotation_matrices = torch.zeros((f.shape[0], 3, 3))
        rotation_matrices[:, :, 2] = normals
        rotation_matrices[:, :, 0] = torch.norm(v[f[:, 0]] - v[f[:, 1]])
        rotation_matrices[:, :, 1] = torch.norm(torch.cross(rotation_matrices[:, :, 2], rotation_matrices[:, :, 0]))

        r = R.from_matrix(rotation_matrices)
        quat = r.as_quat() # (x, y, z, w)
        quat = quat[:, [3, 0, 1, 2]] # (w, x, y, z)

        return quat
    
    def sample_surface_points_with_barycentric(self, v, f, n):
        # M : num vertices, N : num faces, K : num points on a triangle
        bary_weights = self.generate_bary_coords_params(f.shape[0])

        normals =   n[f[:, 0], :].unsqueeze(1).repeat(1, bary_weights.shape[1], 1).reshape(-1, 3) * bary_weights[:, :, 0].reshape(-1, 1) + \
                    n[f[:, 1], :].unsqueeze(1).repeat(1, bary_weights.shape[1], 1).reshape(-1, 3) * bary_weights[:, :, 1].reshape(-1, 1) + \
                    n[f[:, 2], :].unsqueeze(1).repeat(1, bary_weights.shape[1], 1).reshape(-1, 3) * bary_weights[:, :, 2].reshape(-1, 1)
        
        rotations = self.generate_rotations_from_triangles(v, f, normals)

        centers =   v[f[:, 0], :].unsqueeze(1).repeat(1, bary_weights.shape[1], 1).reshape(-1, 3) * bary_weights[:, :, 0].reshape(-1, 1) + \
                    v[f[:, 1], :].unsqueeze(1).repeat(1, bary_weights.shape[1], 1).reshape(-1, 3) * bary_weights[:, :, 1].reshape(-1, 1) + \
                    v[f[:, 2], :].unsqueeze(1).repeat(1, bary_weights.shape[1], 1).reshape(-1, 3) * bary_weights[:, :, 2].reshape(-1, 1)
        
        return centers.reshape(-1, 3), rotations.reshape(-1, 4)
    
    def generate_new_gaussians_features(self, v, f, n):
        centers, rotations = self.sample_surface_points_with_barycentric(v, f, n)
        dist2 = torch.clamp_min(distCUDA2(centers.float().cuda()), 0.0000001)
        scales = torch.sqrt(dist2)[...,None].repeat(1, 3)
        scales[:, :2] *= 2
        scales[:, 2] *= 0.2
        scales = torch.log(scales)

        shs = RGB2SH(np.random.random((centers.shape[0], 3)) / 255.0)
        features = torch.zeros((shs.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = shs
        features[:, 3:, 1:] = 0.0

        opacities = inverse_sigmoid(0.1 * torch.ones((centers.shape[0], 1), dtype=torch.float, device=self.device))

        self._xyz = nn.Parameter(centers.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rotations.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((centers.shape[0]), device=self.device)

    def training_setup_second_stage(self, training_args):
        l = [
                {'params' : [self.grid], 'lr': 2e-4, 'name' : 'grid'},
                {'params': list(self.mesh2gaussians.parameters()), 'lr': 2e-4, "name": "net"}
            ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        # self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args['position_lr_init'] * self.spatial_lr_scale,
        #                                                 lr_final=training_args['position_lr_final'] * self.spatial_lr_scale,
        #                                                 lr_delay_mult=training_args['position_lr_delay_mult'],
        #                                                 max_steps=training_args['position_lr_max_steps'])

        

    def init_second_stage(self, debug_save=False, debug_load=True):
        print ("[INFO] Initializing stage 2...")
        if debug_load:
            density_fields = torch.load('./density_field.pt')
            opacities = self.get_opacity
            mask = (opacities > 0.005).squeeze(1)
            self.resolution = [128, 128, 128]
            xyzs = self.get_xyz[mask]
            
            # normalize to ~ [-1, 1]
            mn, mx = xyzs.amin(0), xyzs.amax(0)
            self.grid_center = (mn + mx) / 2
            self.grid_scale = 1.8 / (mx - mn).amax().item() 
            self.xyz_min = mn
            self.xyz_max = mx
        else:
            density_fields = self.extract_fields()
        if debug_save:
            torch.save(density_fields, './density_field.pt')
        

        # self.grid = DenseGrid(1, self.resolution, self.xyz_min, self.xyz_max)
        # self.grid.update_grid(density_fields[None, None, ...])
        self.grid = nn.Parameter(density_fields.requires_grad_(True))
        self.grid.retain_grad()

        self.grid2mesh = Grid2Mesh.apply

        self.mesh2gaussians = Mesh2GaussiansNetwork().to(self.device)

        self.active_sh_degree = 0
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)


    def update_second_stage(self):
        v, f, n = self.grid2mesh(self.grid)
        # rescale the vertices from [0, resolution] to [-1, 1]
        # v = (v / torch.tensor(self.resolution).float().to(self.device) - 0.5) * 2
        v = 2.0 * (v - 0.5)
        v = v.float()
        f = f.long()
        # v.retain_grad()
        # f.retain_grad()
        # n.retain_grad()
        gaussians = self.mesh2gaussians(v.unsqueeze(0), f.unsqueeze(0), n.unsqueeze(0))

        self._xyz = (gaussians['xyz'].squeeze(0) / self.grid_scale) + self.grid_center

        features = torch.zeros((self._xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().to(self.device)
        features[:, :3, 0 ] = RGB2SH(gaussians['rgb'].squeeze(0))
        features[:, 3:, 1:] = 0.0
        self._features_dc = features[:,:,0:1].transpose(1, 2).contiguous()
        self._features_rest = features[:,:,1:].transpose(1, 2).contiguous()
        self._features_dc.retain_grad()
        self._features_rest.retain_grad()

        self._scaling = self.scaling_inverse_activation(0.1*gaussians['scaling'].squeeze(0))
        # r = R.from_matrix(gaussians['rotation_matrix'])
        # quat = r.as_quat() # (x, y, z, w)
        # quat = quat[:, [3, 0, 1, 2]] # (w, x, y, z)
        self._rotation = matrix_to_quaternion(gaussians['rotation_matrix'].squeeze(0))
        # dummy_rot = torch.eye(3).repeat(self._xyz.shape[0], 1, 1).cuda()
        # self._rotation = matrix_to_quaternion(dummy_rot)
        # self._rotation = gaussians['rotation']
        self._opacity = self.inverse_opacity_activation(gaussians['opacity'].squeeze(0))

        # print (self.get_xyz.shape)
        # print (self.get_scaling.shape)
        # print (self.get_rotation.shape)
        # print (self.get_opacity.shape)
        # print (self.get_features.shape)
        # print (self.grid.shape)

    # def init_pipeline(self, resolution=128, use_flexicubes=False):
    #     self.use_flexicubs = use_flexicubes
    #     if not use_flexicubes:
    #         X = torch.linspace(-1, 1, resolution)
    #         Y = torch.linspace(-1, 1, resolution)
    #         Z = torch.linspace(-1, 1, resolution)
    #         self.grid_coordinate = torch.stack(torch.meshgrid(X, Y, Z, indexing='ij'), dim=-1).view(-1, 3).cuda()
    #         self.gaussian_net = GaussianNetwork()
    #     else:
    #         from flexicubes.flexicubes import FlexiCubes
    #         self.fc = FlexiCubes()
    #         self.grid_coordinate, self.cube_fx8 = self.fc.construct_voxel_grid(resolution)
    #         self.grid_coordinate *= 2
    #         fc_cube_weights_ = torch.zeros((self.cube_fx8.shape[0], 21), dtype=torch.float, device=self.device)
    #         self.fc_cube_weights = torch.nn.Parameter(fc_cube_weights_.clone().detach(), requires_grad=True)

    #         self.gaussian_net = GaussianNetwork(self.fc, self.fc_cube_weights, resolution, self.device)
        
    #     l = [
    #             {'params': list(self.gaussian_net.sdf_grid_net.parameters()), 'lr': 5e-4, "name": "sdf_net"},
    #             {'params': list(self.gaussian_net.gaussian_geo_from_mesh.parameters()), 'lr': 1e-4, "name": "gauss_geo_net"},
    #             {'params': list(self.gaussian_net.gaussian_rgb_from_mesh.parameters()), 'lr': 2e-4, "name": "gauss_rgb_net"},
    #             # {'params': self.fc_cube_weights, 'lr': 2e-4, "name": "fc_weights"}
    #         ]

    #     self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
    #     self.sdf_net_lr_scheduler = get_expon_lr_func(lr_init=5e-4,
    #                                                 lr_final=3e-6,
    #                                                 lr_delay_mult=0.01,
    #                                                 max_steps=30_000)
        
    #     self.gauss_geo_net_lr_scheduler = get_expon_lr_func(lr_init=1e-4,
    #                                                 lr_final=1e-6,
    #                                                 lr_delay_mult=0.01,
    #                                                 max_steps=30_000)
        
    #     self.gauss_rgb_net_lr_scheduler = get_expon_lr_func(lr_init=2e-4,
    #                                                 lr_final=1e-6,
    #                                                 lr_delay_mult=0.01,
    #                                                 max_steps=30_000)

    def init_pipeline(self):
        self.depth_gaussian_net = DepthGaussianNetwork() 
        l = [
                {'params': list(self.depth_gaussian_net.sdf_network.parameters()), 'lr': 5e-4, "name": "sdf_net"},
                {'params': list(self.depth_gaussian_net.deviation_network.parameters()), 'lr': 1e-4, "name": "gauss_geo_net"},
                {'params': list(self.depth_gaussian_net.gaussian_network.parameters()), 'lr': 2e-4, "name": "gauss_rgb_net"},
                # {'params': self.fc_cube_weights, 'lr': 2e-4, "name": "fc_weights"}
            ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.sdf_net_lr_scheduler = get_expon_lr_func(lr_init=5e-4,
                                                    lr_final=3e-6,
                                                    lr_delay_mult=0.01,
                                                    max_steps=30_000)
        
        self.gauss_geo_net_lr_scheduler = get_expon_lr_func(lr_init=1e-4,
                                                    lr_final=1e-6,
                                                    lr_delay_mult=0.01,
                                                    max_steps=30_000)
        
        self.gauss_rgb_net_lr_scheduler = get_expon_lr_func(lr_init=2e-4,
                                                    lr_final=1e-6,
                                                    lr_delay_mult=0.01,
                                                    max_steps=30_000)

    # def run_pipeline(self, cam_pos):
    #     if self.use_flexicubs:
    #         gaussians, ret = self.gaussian_net(self.grid_coordinate, cam_pos, self.cube_fx8)
    #     else:
    #         gaussians, ret = self.gaussian_net(self.grid_coordinate, cam_pos)
        
    #     self._xyz = gaussians['xyz'].squeeze(0)
    #     features = torch.zeros((self._xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().to(self.device)
    #     features[:, :3, 0 ] = RGB2SH(gaussians['rgb'].squeeze(0))
    #     features[:, 3:, 1:] = 0.0
    #     self._features_dc = features[:,:,0:1].transpose(1, 2).contiguous()
    #     self._features_rest = features[:,:,1:].transpose(1, 2).contiguous()
    #     self._features_dc.retain_grad()
    #     self._features_rest.retain_grad()

    #     self._scaling = self.scaling_inverse_activation(gaussians['scaling'].squeeze(0))
 
    #     self._rotation = matrix_to_quaternion(gaussians['rotation_matrix'].squeeze(0))
 
    #     self._opacity = self.inverse_opacity_activation(gaussians['opacity'].squeeze(0))

    #     # print (self.get_xyz.shape)
    #     # print (self.get_scaling.shape)
    #     # print (self.get_rotation.shape)
    #     # print (self.get_opacity.shape)
    #     # print (self.get_features.shape)
        
    #     # if self.use_flexicubs:
    #     #     return v.clone().detach().cpu().numpy(), f.clone().detach().cpu().numpy(), l, grad
    #     # else:
    #     #     return v.clone().detach().cpu().numpy(), f.clone().detach().cpu().numpy(), n.clone().detach().cpu().numpy(), grad
    #     return ret

    def run_pipeline(self, rays_o, rays_d):
        ret = self.depth_gaussian_net(rays_o, rays_d)
        gaussians = ret['gaussians']
        gradients = ret['gradients']

        self._xyz = gaussians['xyz'].squeeze(0)
        features = torch.zeros((self._xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().to(self.device)
        features[:, :3, 0 ] = RGB2SH(gaussians['rgb'].squeeze(0))
        features[:, 3:, 1:] = 0.0
        self._features_dc = features[:,:,0:1].transpose(1, 2).contiguous()
        self._features_rest = features[:,:,1:].transpose(1, 2).contiguous()
        self._features_dc.retain_grad()
        self._features_rest.retain_grad()

        self._scaling = self.scaling_inverse_activation(gaussians['scaling'].squeeze(0))
 
        self._rotation = matrix_to_quaternion(gaussians['rotation_matrix'].squeeze(0))
 
        self._opacity = self.inverse_opacity_activation(gaussians['opacity'].squeeze(0))

        return gradients

    def update_gaussians_from_batch(self, gauss_batch):
        self._xyz = gauss_batch['xyz'].squeeze(0)
        features = torch.zeros((self._xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().to(self.device)
        features[:, :3, 0 ] = RGB2SH(gauss_batch['rgb'].squeeze(0))
        features[:, 3:, 1:] = 0.0
        self._features_dc = features[:,:,0:1].transpose(1, 2).contiguous()
        self._features_rest = features[:,:,1:].transpose(1, 2).contiguous()
        self._features_dc.retain_grad()
        self._features_rest.retain_grad()

        self._scaling = self.scaling_inverse_activation(gauss_batch['scaling'].squeeze(0))
 
        self._rotation = matrix_to_quaternion(gauss_batch['rotation_matrix'].squeeze(0))
 
        self._opacity = self.inverse_opacity_activation(gauss_batch['opacity'].squeeze(0))








