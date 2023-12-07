import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from simple_knn._C import distCUDA2

from skimage import measure
from scipy.spatial.transform import Rotation as R

from utils.general_utils import inverse_sigmoid, get_expon_lr_func, strip_symmetric, build_scaling_rotation
from utils.sh_utils import eval_sh, SH2RGB, RGB2SH

''' Dense 3D grid
'''
class DenseGrid(nn.Module):
    def __init__(self, channels, world_size, xyz_min, xyz_max, **kwargs):
        super(DenseGrid, self).__init__()
        self.channels = channels
        self.world_size = world_size
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.grid = nn.Parameter(torch.zeros([1, channels, *world_size]))

    def assign_grid(self, in_grid, xyz_min, xyz_max):
        # in_grid shape: (1, channels, res_x, res_y, res_z)
        self.channels = in_grid.shape[1]
        self.world_size = in_grid.shape[2:]
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.grid = nn.Parameter(in_grid.requires_grad_(True))

    def update_grid(self, in_grid):
        self.grid = nn.Parameter(in_grid.requires_grad_(True))


    def forward(self, xyz):
        '''
        xyz: global coordinates to query
        '''
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,1,-1,3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        out = F.grid_sample(self.grid, ind_norm, mode='bilinear', align_corners=True)
        out = out.reshape(self.channels,-1).T.reshape(*shape,self.channels)
        if self.channels == 1:
            out = out.squeeze(-1)
        return out

    def scale_volume_grid(self, new_world_size):
        if self.channels == 0:
            self.grid = nn.Parameter(torch.zeros([1, self.channels, *new_world_size]))
        else:
            self.grid = nn.Parameter(
                F.interpolate(self.grid.data, size=tuple(new_world_size), mode='trilinear', align_corners=True))

    # def total_variation_add_grad(self, wx, wy, wz, dense_mode):
    #     '''Add gradients by total variation loss in-place'''
    #     total_variation_cuda.total_variation_add_grad(
    #         self.grid, self.grid.grad, wx, wy, wz, dense_mode)

    def get_dense_grid(self):
        return self.grid

    @torch.no_grad()
    def __isub__(self, val):
        self.grid.data -= val
        return self

    def extra_repr(self):
        return f'channels={self.channels}, world_size={self.world_size.tolist()}'

def scatter_to_grid(inds, vals, size):
    """
    Scatter update values into empty tensor of size size.
    :param inds: (#values, dims)
    :param vals: (#values)
    :param size: tuple for size. len(size)=dims
    """
    dims = inds.shape[1]
    assert(inds.shape[0] == vals.shape[0])
    assert(len(size) == dims)
    dev = vals.device
    # result = torch.zeros(*size).view(-1).to(dev).type(vals.dtype)  # flatten
    # # flatten inds
    result = torch.zeros(*size, device=dev).view(-1).type(vals.dtype)  # flatten
    # flatten inds
    fac = [np.prod(size[i+1:]) for i in range(len(size)-1)] + [1]
    fac = torch.tensor(fac, device=dev).type(inds.dtype)
    inds_fold = torch.sum(inds*fac, dim=-1)  # [#values,]
    result.scatter_add_(0, inds_fold, vals)
    result = result.view(*size)
    return result

def point_rasterize(pts, vals, size):
    """
    :param pts: point coords, tensor of shape (batch, num_points, dim) within range (0, 1)
    :param vals: point values, tensor of shape (batch, num_points, features)
    :param size: len(size)=dim tuple for grid size
    :return rasterized values (batch, features, res0, res1, res2)
    """
    dim = pts.shape[-1]
    assert(pts.shape[:2] == vals.shape[:2])
    assert(pts.shape[2] == dim)
    size_list = list(size)
    size = torch.tensor(size).to(pts.device).float()
    cubesize = 1.0 / size
    bs = pts.shape[0]
    nf = vals.shape[-1]
    npts = pts.shape[1]
    dev = pts.device
    
    ind0 = torch.floor(pts / cubesize).long()  # (batch, num_points, dim)
    ind1 = torch.fmod(torch.ceil(pts / cubesize), size).long() # periodic wrap-around
    ind01 = torch.stack((ind0, ind1), dim=0) # (2, batch, num_points, dim)
    tmp = torch.tensor([0,1],dtype=torch.long)
    com_ = torch.stack(torch.meshgrid(tuple([tmp] * dim)), dim=-1).view(-1, dim)
    dim_ = torch.arange(dim).repeat(com_.shape[0], 1) # (2**dim, dim)
    ind_ = ind01[com_, ..., dim_]   # (2**dim, dim, batch, num_points)
    ind_n = ind_.permute(2, 3, 0, 1) # (batch, num_points, 2**dim, dim)
    # ind_b = torch.arange(bs).expand(ind_n.shape[1], ind_n.shape[2], bs).permute(2, 0, 1) # (batch, num_points, 2**dim)
    ind_b = torch.arange(bs, device=dev).expand(ind_n.shape[1], ind_n.shape[2], bs).permute(2, 0, 1) # (batch, num_points, 2**dim)
    
    # weights of neighboring nodes
    xyz0 = ind0.type(cubesize.dtype) * cubesize        # (batch, num_points, dim)
    xyz1 = (ind0.type(cubesize.dtype) + 1) * cubesize  # (batch, num_points, dim)
    xyz01 = torch.stack((xyz0, xyz1), dim=0) # (2, batch, num_points, dim)
    pos = xyz01[com_, ..., dim_].permute(2,3,0,1)   # (batch, num_points, 2**dim, dim)
    pos_ = xyz01[1-com_, ..., dim_].permute(2,3,0,1)   # (batch, num_points, 2**dim, dim)
    pos_ = pos_.type(pts.dtype)
    dxyz_ = torch.abs(pts.unsqueeze(-2) - pos_) / cubesize # (batch, num_points, 2**dim, dim)
    weights = torch.prod(dxyz_, dim=-1, keepdim=False) # (batch, num_points, 2**dim)
    
    ind_b = ind_b.unsqueeze(-1).unsqueeze(-1)      # (batch, num_points, 2**dim, 1, 1)
    ind_n = ind_n.unsqueeze(-2)                    # (batch, num_points, 2**dim, 1, dim)
    ind_f = torch.arange(nf, device=dev).view(1, 1, 1, nf, 1)  # (1, 1, 1, nf, 1)
    # ind_f = torch.arange(nf).view(1, 1, 1, nf, 1)  # (1, 1, 1, nf, 1)
    
    ind_b = ind_b.expand(bs, npts, 2**dim, nf, 1)
    ind_n = ind_n.expand(bs, npts, 2**dim, nf, dim).to(dev)
    ind_f = ind_f.expand(bs, npts, 2**dim, nf, 1)
    inds = torch.cat([ind_b, ind_f, ind_n], dim=-1)  # (batch, num_points, 2**dim, nf, 1+1+dim)
     
    # weighted values
    vals = weights.unsqueeze(-1) * vals.unsqueeze(-2)   # (batch, num_points, 2**dim, nf)
    
    inds = inds.view(-1, dim+2).permute(1, 0).long()  # (1+dim+1, bs*npts*2**dim*nf)
    vals = vals.reshape(-1) # (bs*npts*2**dim*nf)
    tensor_size = [bs, nf] + size_list
    # print(inds)
    # print(vals)
    
    raster = scatter_to_grid(inds.permute(1, 0), vals, tensor_size)
    
    return raster  # [batch, nf, res, res, res]

class Grid2Mesh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grid):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """

        grid_numpy = grid.detach().cpu().numpy()
        verts, faces, normals, values = measure.marching_cubes(grid_numpy, level=0.5)
        
        # verts /= grid_numpy.shape[-3:] # rescale the vertices from the resolution to [0, 1]
        # rescale the vertices from [0, resolution] to [-1, 1]
        verts = (verts / grid_numpy.shape[-3:] - 0.5) * 2
        device = grid.device
        verts = torch.Tensor(np.ascontiguousarray(verts)).to(device)
        faces = torch.Tensor(np.ascontiguousarray(faces)).to(device)
        normals = torch.Tensor(np.ascontiguousarray(-normals)).to(device)

        res = torch.tensor(grid.detach().shape[-1])
        ctx.save_for_backward(verts, normals, res)

        return verts, faces, normals

    @staticmethod
    def backward(ctx, dL_dVertex, dL_dFace, dL_dNormals):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        
        vert_pts, normals, res = ctx.saved_tensors

        res = (res.item(), res.item(), res.item())
        # matrix multiplication between dL/dV and dV/dPSR
        # dV/dPSR = - normals
        # grad_vert = torch.matmul(dL_dVertex.permute(1, 0, 2), -normals.permute(1, 2, 0))
        grad_vert = torch.matmul(dL_dVertex.unsqueeze(1), -normals.unsqueeze(2))
        grad_grid = point_rasterize(vert_pts.unsqueeze(0), grad_vert.permute(1, 0, 2), res) # b x 1 x res x res x res
        
        return grad_grid
    
# def gaussian_3d_coeff(xyzs, covs):
#     # xyzs: [N, 3] : query_xyzs - gaussian_center_xyzs
#     # covs: [N, 6]
#     x, y, z = xyzs[:, 0], xyzs[:, 1], xyzs[:, 2]
#     a, b, c, d, e, f = covs[:, 0], covs[:, 1], covs[:, 2], covs[:, 3], covs[:, 4], covs[:, 5]

#     # eps must be small enough !!!
#     inv_det = 1 / (a * d * f + 2 * e * c * b - e**2 * a - c**2 * d - b**2 * f + 1e-24)
#     inv_a = (d * f - e**2) * inv_det
#     inv_b = (e * c - b * f) * inv_det
#     inv_c = (e * b - c * d) * inv_det
#     inv_d = (a * f - c**2) * inv_det
#     inv_e = (b * c - e * a) * inv_det
#     inv_f = (a * d - b**2) * inv_det

#     power = -0.5 * (x**2 * inv_a + y**2 * inv_d + z**2 * inv_f) - x * y * inv_b - x * z * inv_c - y * z * inv_e

#     power[power > 0] = -1e10 # abnormal values... make weights 0
        
#     return torch.exp(power)


# @torch.no_grad()
# def extract_fields(gaussians, resolution=128, num_blocks=16, relax_ratio=1.5):
#     #TODO: get in gaussians and calculate a coarse density grid
#     print("Extracting density fields...")
        
#     block_size = 2 / num_blocks # 2 : -1 ~ 1

#     assert resolution % block_size == 0
#     split_size = resolution // num_blocks

#     opacities = gaussians.get_opacity

#     # pre-filter low opacity gaussians to save computation
#     mask = (opacities > 0.005).squeeze(1)

#     opacities = opacities[mask]
#     xyzs = gaussians.get_xyz[mask]
#     stds = gaussians.get_scaling[mask]
    
#     # normalize to ~ [-1, 1]
#     mn, mx = xyzs.amin(0), xyzs.amax(0)
#     gaussians.grid_center = (mn + mx) / 2
#     gaussians.grid_scale = 1.8 / (mx - mn).amax().item() 

#     xyzs = (xyzs - gaussians.grid_center) * gaussians.grid_scale
#     stds = stds * gaussians.grid_scale

#     covs = gaussians.covariance_activation(stds, 1, gaussians._rotation[mask]) # symmetric matrix with 6 parameters for each gaussian

#     # tile
#     device = opacities.device
#     occ = torch.zeros([resolution] * 3, dtype=torch.float32, device=device)

#     X = torch.linspace(-1, 1, resolution).split(split_size)
#     Y = torch.linspace(-1, 1, resolution).split(split_size)
#     Z = torch.linspace(-1, 1, resolution).split(split_size)

#     # loop blocks (assume max size of gaussian is small than relax_ratio * block_size !!!)
#     for xi, xs in enumerate(X):
#         for yi, ys in enumerate(Y):
#             for zi, zs in enumerate(Z):
#                 # sample fine grid points inside each coarse voxel
#                 xx, yy, zz = torch.meshgrid(xs, ys, zs)
#                 # sample points [M, 3] M : number of query points inside this coarse voxel
#                 pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).to(device)
#                 # in-tile gaussians mask
#                 vmin, vmax = pts.amin(0), pts.amax(0)
#                 vmin -= block_size * relax_ratio
#                 vmax += block_size * relax_ratio
#                 mask = (xyzs < vmax).all(-1) & (xyzs > vmin).all(-1)
#                 # if hit no gaussian, continue to next block
#                 if not mask.any():
#                     continue
#                 # L : number of Gaussians inside this coarse voxel
#                 mask_xyzs = xyzs[mask] # [L, 3]
#                 mask_covs = covs[mask] # [L, 6]
#                 mask_opas = opacities[mask].view(1, -1) # [L, 1] --> [1, L]

#                 # query per point-gaussian pair.
#                 # The distance between each query point (M) and each gaussian center (L)
#                 g_pts = pts.unsqueeze(1).repeat(1, mask_covs.shape[0], 1) - mask_xyzs.unsqueeze(0) # [M, L, 3]
#                 g_covs = mask_covs.unsqueeze(0).repeat(pts.shape[0], 1, 1) # [M, L, 6]

#                 # batch on gaussian to avoid OOM
#                 batch_g = 1024
#                 val = 0
#                 for start in range(0, g_covs.shape[1], batch_g):
#                     end = min(start + batch_g, g_covs.shape[1])
#                     w = gaussian_3d_coeff(g_pts[:, start:end].reshape(-1, 3), g_covs[:, start:end].reshape(-1, 6)).reshape(pts.shape[0], -1) # [M, l]
#                     val += (mask_opas[:, start:end] * w).sum(-1)
                
#                 occ[xi * split_size: xi * split_size + len(xs), 
#                     yi * split_size: yi * split_size + len(ys), 
#                     zi * split_size: zi * split_size + len(zs)] = val.reshape(len(xs), len(ys), len(zs)) 

#     return occ

# def generate_bary_coords_params(num_faces):
#     # [K, 3] K : number of sampled points on a triangle
#     default_barycentric_weights = torch.tensor([
#                                                 [1./3, 1./3, 1./3], 
#                                                 [1./2, 1./4, 1./4], 
#                                                 [1./4, 1./2, 1./4],
#                                                 [1./4, 1./4, 1./2]])
    

#     new_centers_bary_weights = default_barycentric_weights.unsqueeze(0).repeat(num_faces, 1, 1) # [N, K, 3]
#     # new_centers_bary_weights = default_barycentric_weights.reshape(-1, 3)
    
#     # return nn.Paramter(torch.tensor(new_centers_bary_weights).requires_grad_(True)) 
#     print (new_centers_bary_weights.shape)
#     return new_centers_bary_weights

# def generate_rotation(v, f, normals):
#     # Last column of the rotation matrix is the triangle surface normal
#     # one column is an edge of the triangle
#     # the other one is compyted by cross product
#     rotation_matrices = torch.zeros((f.shape[0], 3, 3))
#     rotation_matrices[:, :, 2] = normals
#     rotation_matrices[:, :, 0] = torch.norm(v[f[:, 0]] - v[f[:, 1]])
#     rotation_matrices[:, :, 1] = torch.norm(torch.cross(rotation_matrices[:, :, 2], rotation_matrices[:, :, 0]))

#     r = R.from_matrix(rotation_matrices)
#     quat = r.as_quat() # (x, y, z, w)
#     quat = quat[:, [3, 0, 1, 2]] # (w, x, y, z)

#     return quat

# def sample_surface_points_with_barycentric(v, f, n):
#     # M : num vertices, N : num faces, K : num points on a triangle
#     bary_weights = generate_bary_coords_params(f.shape[0])

#     normals =   n[f[:, 0], :].unsqueeze(1).repeat(1, bary_weights.shape[1], 1).reshape(-1, 3) * bary_weights[:, :, 0].reshape(-1, 1) + \
#                 n[f[:, 1], :].unsqueeze(1).repeat(1, bary_weights.shape[1], 1).reshape(-1, 3) * bary_weights[:, :, 1].reshape(-1, 1) + \
#                 n[f[:, 2], :].unsqueeze(1).repeat(1, bary_weights.shape[1], 1).reshape(-1, 3) * bary_weights[:, :, 2].reshape(-1, 1)
    
#     rotations = generate_rotation(v, f, normals)

#     centers =   v[f[:, 0], :].unsqueeze(1).repeat(1, bary_weights.shape[1], 1).reshape(-1, 3) * bary_weights[:, :, 0].reshape(-1, 1) + \
#                 v[f[:, 1], :].unsqueeze(1).repeat(1, bary_weights.shape[1], 1).reshape(-1, 3) * bary_weights[:, :, 1].reshape(-1, 1) + \
#                 v[f[:, 2], :].unsqueeze(1).repeat(1, bary_weights.shape[1], 1).reshape(-1, 3) * bary_weights[:, :, 2].reshape(-1, 1)
    
#     return centers.reshape(-1, 3), rotations.reshape(-1, 4)

# def generate_new_gaussians_features(gaussians, v, f, n):
#     centers, rotations = sample_surface_points_with_barycentric(v, f, n)
#     dist2 = torch.clamp_min(distCUDA2(centers.float().cuda()), 0.0000001)
#     scales = torch.sqrt(dist2)[...,None].repeat(1, 3)
#     scales[:, :2] *= 2
#     scales[:, 2] *= 0.2
#     scales = torch.log(scales)

#     shs = RGB2SH(np.random.random((centers.shape[0], 3)) / 255.0)
#     features = torch.zeros((shs.shape[0], 3, (gaussians.max_sh_degree + 1) ** 2)).float().cuda()
#     features[:, :3, 0 ] = shs
#     features[:, 3:, 1:] = 0.0

#     opacities = inverse_sigmoid(0.1 * torch.ones((centers.shape[0], 1), dtype=torch.float, device="cuda"))

#     gaussians._xyz = nn.Parameter(centers.requires_grad_(True))
#     gaussians._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
#     gaussians._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
#     gaussians._scaling = nn.Parameter(scales.requires_grad_(True))
#     gaussians._rotation = nn.Parameter(rotations.requires_grad_(True))
#     gaussians._opacity = nn.Parameter(opacities.requires_grad_(True))
#     gaussians.max_radii2D = torch.zeros((centers.shape[0]), device="cuda")


# def training_setup(grid, bary_weights, gaussians, training_args):
#     l = [
#             {'params' : grid, 'lr': training_args['position_lr_init'], 'name' : 'grid'},
#             {'params' : bary_weights, 'lr': training_args['feature_lr'], 'name' : 'bary_weights'},
#             {'params': [gaussians._features_dc], 'lr': training_args['feature_lr'], "name": "f_dc"},
#             {'params': [gaussians._features_rest], 'lr': training_args['feature_lr'] / 20.0, "name": "f_rest"},
#             {'params': [gaussians._opacity], 'lr': training_args['opacity_lr'], "name": "opacity"},
#             {'params': [gaussians._scaling], 'lr': training_args['scaling_lr'], "name": "scaling"},
#             {'params': [gaussians._rotation], 'lr': training_args['rotation_lr'], "name": "rotation"}
#         ]

#     optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
#     gaussians.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args['position_lr_init'] * gaussians.spatial_lr_scale,
#                                                     lr_final=training_args['position_lr_final'] * gaussians.spatial_lr_scale,
#                                                     lr_delay_mult=training_args['position_lr_delay_mult'],
#                                                     max_steps=training_args['position_lr_max_steps'])


# # def gaussians_from_grid(grid : DenseGrid):
# #     grid2mesh = Grid2Mesh.apply

# #     v, f, n = grid2mesh(grid) 

# #     centers, scales, rotations = sample_surface_points_with_barycentric(v, f, n)

# #     features = generate_new_gaussians_features(gaussians, v, f, n)

# #     training_setup()

# # class Grid2Gaussians(nn.Module):
# #     def setup_functions(self):
# #         def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
# #             L = build_scaling_rotation(scaling_modifier * scaling, rotation)
# #             actual_covariance = L @ L.transpose(1, 2)
# #             symm = strip_symmetric(actual_covariance)
# #             return symm
        
# #         self.scaling_activation = torch.exp
# #         self.scaling_inverse_activation = torch.log

# #         self.covariance_activation = build_covariance_from_scaling_rotation

# #         self.opacity_activation = torch.sigmoid
# #         self.inverse_opacity_activation = inverse_sigmoid

# #         self.rotation_activation = torch.nn.functional.normalize
    
# #     def __init__(self):
# #         grid2mesh = Grid2Mesh.apply

# #         self._grid = torch.empty(0)
# #         self._xyz = torch.empty(0)
# #         self._feature_dc = torch.empty(0)
# #         self._features_rest = torch.empty(0)
# #         self._scaling = torch.empty(0)
# #         self._rotation = torch.empty(0)
# #         self._opacity = torch.empty(0)
# #         self.max_radii2D = torch.empty(0)
# #         self.xyz_gradient_accum = torch.empty(0)
# #         self.denom = torch.empty(0)
# #         self.optimizer = None
# #         self.percent_dense = 0
# #         self.spatial_lr_scale = 0
# #         self.setup_functions()


# # def train_demo():
# #     gaussians = precompute_gaussians()
    
# #     field = extract_field(gaussians)

# #     del gaussians
    
# #     grid = DenseGrid()
# #     grid.assign(field)

# #     gaussians_from_grid(grid)

# #     for i in iterations:
# #         view = select_view()

# #         rendered = render_gaussians(view, gaussians)

# #         loss += photo_loss(rendered, gt)

# #         loss += gradient_regularization_loss()

# #         loss.backward()

# #         with torch.no_grad():
# #             optimizer.step()
# #             optimizer.zero_grad()



















    

