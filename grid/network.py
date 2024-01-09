import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from grid.embedder import *
from utils.rigid_utils import exp_so3
from grid.grid import Grid2Mesh, DenseGrid

class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            sdf_bounding_sphere,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0,
            sphere_scale=1.0,
            inside_outside=False,
    ):
        super().__init__()

        self.sdf_bounding_sphere = sdf_bounding_sphere
        self.sphere_scale = sphere_scale
        dims = [d_in] + dims + [d_out + feature_vector_size]

        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            dims[0] = input_ch
        print(multires, dims)
        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)

                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)

    def forward(self, input):
        if self.embed_fn is not None:
            input = self.embed_fn(input)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        return x

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[:,:1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients

    def get_outputs(self, x):
        x.requires_grad_(True)
        output = self.forward(x)
        sdf = output[:,:1]
        ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf)
        feature_vectors = output[:, 1:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        return sdf, feature_vectors, gradients

    def get_sdf_vals(self, x):
        sdf = self.forward(x)[:,:1]
        ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf)
        return sdf


from hashencoder.hashgrid import _hash_encode, HashEncoder
class ImplicitNetworkGrid(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            sdf_bounding_sphere,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0,
            sphere_scale=1.0,
            inside_outside=False,
            base_size = 16, #16,
            end_size = 2048,#2048,
            logmap = 19,#19,
            num_levels=16,#16,
            level_dim=2,
            divide_factor = 1.5, # used to normalize the points range for multi-res grid
            use_grid_feature = True
    ):
        super().__init__()

        self.sdf_bounding_sphere = sdf_bounding_sphere
        self.sphere_scale = sphere_scale
        dims = [d_in] + dims + [d_out + feature_vector_size]
        self.embed_fn = None
        self.divide_factor = divide_factor
        self.grid_feature_dim = num_levels * level_dim
        self.use_grid_feature = use_grid_feature
        dims[0] += self.grid_feature_dim
        
        print(f"using hash encoder with {num_levels} levels, each level with feature dim {level_dim}")
        print(f"resolution:{base_size} -> {end_size} with hash map size {logmap}")
        self.encoding = HashEncoder(input_dim=3, num_levels=num_levels, level_dim=level_dim, 
                    per_level_scale=2, base_resolution=base_size, 
                    log2_hashmap_size=logmap, desired_resolution=end_size)
        
        '''
        # can also use tcnn for multi-res grid as it now supports eikonal loss
        base_size = 16
        hash = True
        smoothstep = True
        self.encoding = tcnn.Encoding(3, {
                        "otype": "HashGrid" if hash else "DenseGrid",
                        "n_levels": 16,
                        "n_features_per_level": 2,
                        "log2_hashmap_size": 19,
                        "base_resolution": base_size,
                        "per_level_scale": 1.34,
                        "interpolation": "Smoothstep" if smoothstep else "Linear"
                    })
        '''
        
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            dims[0] += input_ch - 3
            print ("input_ch", input_ch)
        print("network architecture")
        print(dims)
        
        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)

                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)
        self.cache_sdf = None

    def get_encoding(self, input):
        feature = self.encoding(input / self.divide_factor)
        embed = self.embed_fn(input)
        encoded = torch.cat((embed, feature), dim=-1)
        return encoded

    def forward(self, input):
        if self.use_grid_feature:
            # normalize point range as encoding assume points are in [-1, 1]
            feature = self.encoding(input / self.divide_factor)
        else:
            feature = torch.zeros_like(input[:, :1].repeat(1, self.grid_feature_dim))
                    
        if self.embed_fn is not None:
            embed = self.embed_fn(input)
            input = torch.cat((embed, feature), dim=-1)
        else:
            input = torch.cat((input, feature), dim=-1)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        return x

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[:,:1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients

    def get_outputs(self, x):
        x.requires_grad_(True)
        output = self.forward(x)
        sdf = output[:,:1]

        feature_vectors = output[:, 1:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        return sdf, feature_vectors, gradients

    def get_sdf_vals(self, x):
        sdf = self.forward(x)[:,:1]
        return sdf

    def mlp_parameters(self):
        parameters = []
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            parameters += list(lin.parameters())
        return parameters

    def grid_parameters(self):
        print("grid parameters", len(list(self.encoding.parameters())))
        for p in self.encoding.parameters():
            print(p.shape)
        return self.encoding.parameters()


class Mesh2GaussiansNetwork(nn.Module):
    def __init__(self, L=4, D=128, multires=6):
        super(Mesh2GaussiansNetwork, self).__init__()
        self.L = L
        self.D = D
        self.skips = [L // 2]

        self.embed_fn, vertex_input_ch = get_embedder(multires, 3)

        self.input_ch = 3 * (vertex_input_ch + 16) + 3 # (3 embedded vertices + 3 normals)

        self.linear = nn.ModuleList(
            [nn.Linear(self.input_ch, D)] + [
                nn.Linear(D, D) if i not in self.skips else nn.Linear(D + self.input_ch, D)
                for i in range(D - 1)
            ]
        )

        self.bary_coords_lin = nn.Linear(D, 2)
        self.opacity_lin = nn.Linear(D, 1)
        self.scaling_lin = nn.Linear(D, 3)
        self.rotation_angle_lin = nn.Linear(D, 1)
        # self.rotation_quat_lin = nn.Linear(D, 4)
        self.rgb_lin = nn.Linear(D, 3)

    def forward(self, embeded, vertices, faces):
        vec1 = vertices[:, faces[:, :, 0].flatten()] - vertices[:, faces[:, :, 1].flatten()] 
        vec2 = vertices[:, faces[:, :, 0].flatten()] - vertices[:, faces[:, :, 2].flatten()] 
        cross_pro = torch.cross(vec1, vec2, dim=-1)
        base_scale = torch.sqrt(cross_pro.norm() / torch.pi)
        faces_normal = F.normalize(cross_pro)
        # faces_normal = F.normalize(torch.cross(vec1, vec2, dim=-1))
        x = torch.cat([embeded[:, faces[:, :, 0].flatten()], embeded[:, faces[:, :, 1].flatten()], embeded[:, faces[:, :, 2].flatten()], faces_normal], dim=-1) 
        
        h = self.linear[0](x)

        h = F.relu(h)
        for i in range(1, self.L):
            h = self.linear[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x, h], dim=-1)
        
        bary_coords_2 = torch.sigmoid(self.bary_coords_lin(h))
        bary_coord_last = 1. - torch.sum(bary_coords_2, dim=-1)

        ret = dict()

        ret['xyz'] =  vertices[:, faces[:, :, 0].flatten()] * bary_coords_2[:, :, 0].reshape(-1, 1).unsqueeze(0) + \
                    vertices[:, faces[:, :, 1].flatten()] * bary_coords_2[:, :, 1].reshape(-1, 1).unsqueeze(0) + \
                    vertices[:, faces[:, :, 2].flatten()] * bary_coord_last.reshape(-1, 1).unsqueeze(0)
        
        ret['opacity'] = torch.sigmoid(self.opacity_lin(h))

        ret['scaling']= torch.sigmoid(self.scaling_lin(h)) * base_scale# need the base scaling
        rotation_theta = torch.sigmoid(self.rotation_angle_lin(h)) * 2 * torch.pi
        
        ret['rotation_matrix'] = exp_so3(faces_normal.squeeze(0), rotation_theta.squeeze(0)).unsqueeze(0)
        
        ret['rgb'] = torch.sigmoid(self.rgb_lin(h))

        return ret
    
class GaussianGeoNetwork(nn.Module):
    def __init__(self, D=64, num_geo_feat=16, num_hash_feat=32, num_sdf_feat=16, multires_geo=6, device='cuda'):
        super(GaussianGeoNetwork, self).__init__()

        self.device = device
        self.view_embed_fn, geo_input_ch = get_embedder(multires_geo, 3)
        self.input_ch = 3 * (geo_input_ch + num_hash_feat + num_sdf_feat) + 3 # (3 embedded vertices + 3 normals)

        self.linear = nn.ModuleList(
            [
                nn.Linear(self.input_ch, D),
                nn.Linear(D, D),
            ]
        )

        self.bary_coords_lin = nn.Linear(D, 3)
        self.opacity_lin = nn.Linear(D, 1)
        self.scaling_lin = nn.Linear(D, 3)
        self.rotation_angle_lin = nn.Linear(D, 1)
        self.geo_feat_lin = nn.Linear(D, num_geo_feat)

    def forward(self, embeded, vertices, faces):
        vec1 = vertices[:, faces[:, :, 0].flatten()] - vertices[:, faces[:, :, 1].flatten()] 
        vec2 = vertices[:, faces[:, :, 0].flatten()] - vertices[:, faces[:, :, 2].flatten()] 
        cross_pro = torch.cross(vec1, vec2, dim=-1)
        # base_scale = torch.sqrt(cross_pro.norm() / torch.pi)
        faces_normal = F.normalize(cross_pro)
        # faces_normal = F.normalize(torch.cross(vec1, vec2, dim=-1))
        x = torch.cat([embeded[:, faces[:, :, 0].flatten()], embeded[:, faces[:, :, 1].flatten()], embeded[:, faces[:, :, 2].flatten()], faces_normal], dim=-1) 
        
        for l in self.linear:
            x = l(x)
            x = F.relu(x)

        bary_coords = torch.sigmoid(self.bary_coords_lin(x))
        bary_coords = bary_coords / torch.sum(bary_coords, dim=-1, keepdim=True)

        gaussians = dict()

        gaussians['xyz'] =  vertices[:, faces[:, :, 0].flatten()] * bary_coords[:, :, 0].reshape(-1, 1).unsqueeze(0) + \
                        vertices[:, faces[:, :, 0].flatten()] * bary_coords[:, :, 1].reshape(-1, 1).unsqueeze(0) + \
                        vertices[:, faces[:, :, 0].flatten()] * bary_coords[:, :, 2].reshape(-1, 1).unsqueeze(0)
        
        # geo_feat = self.opacity_lin(x)
        gaussians['opacity'] = torch.sigmoid(self.opacity_lin(x))

        gaussians['scaling']= torch.sigmoid(self.scaling_lin(x)) * (2. / 100) #* base_scale# need the base scaling
        rotation_theta = torch.sigmoid(self.rotation_angle_lin(x)) * 2 * torch.pi
        
        gaussians['rotation_matrix'] = exp_so3(faces_normal.squeeze(0), rotation_theta.squeeze(0)).unsqueeze(0)

        geo_feat = F.relu(self.geo_feat_lin(x))

        return gaussians, geo_feat
        

class GaussianRGBNetwork(nn.Module):
    def __init__(self, D=64, num_geo_feat=16, num_hash_feat=32, multires_view=6, device='cuda'):
        super(GaussianRGBNetwork, self).__init__()

        self.device = device
        self.view_embed_fn, view_ch = get_embedder(multires_view, 3)
        self.input_ch = (view_ch + num_hash_feat) + view_ch + num_geo_feat

        self.linear = nn.ModuleList(
            [
                nn.Linear(self.input_ch, D),
                nn.Linear(D, D),
            ]
        )

        self.rgb = nn.Linear(D, 3)

    def forward(self, vertex_encoded, geo_feature, view_dir):
        view_encoded = self.view_embed_fn(view_dir)
        x = torch.cat([vertex_encoded, view_encoded, geo_feature], dim=-1)

        for l in self.linear:
            x = l(x)
            x = F.relu(x)

        x = torch.sigmoid(self.rgb(x))

        return x
    
class GaussianNetwork_(nn.Module):
    def __init__(self, fc=None, fc_weights=None, resolution=128, device='cuda'):
        super(GaussianNetwork_, self).__init__()

        self.device = device
        self.res = resolution
        self.multires = 6
        self.sdf_grid_net = ImplicitNetworkGrid(16, True, 3, 1, [64, 64], multires=self.multires, divide_factor=1.0).to(self.device)
        self.use_flexicubs = False
        if fc is not None:
            self.use_flexicubs = True
            self.mesh_from_grid = fc
            self.fc_weights = fc_weights
        else:
            self.mesh_from_grid = Grid2Mesh.apply
        self.gaussians_from_mesh = Mesh2GaussiansNetwork(multires=self.multires).to(self.device)

    def forward(self, input, cube_fx8=None):
        # Input is the regular grid with shape (res * res * res, 3)
        grid = self.sdf_grid_net(input)
        sdf_grid = grid[:, :1]
        features_grid = grid[:, 1:]

        if self.use_flexicubs:
            v, f, l = self.mesh_from_grid(input, sdf_grid, cube_fx8, self.res, 
                                          self.fc_weights[:, :12], self.fc_weights[:, 12:20], self.fc_weights[:, 20], True, False, None)
        else:
            v, f, n = self.mesh_from_grid(sdf_grid.view(self.res, self.res, self.res), 0.0)
            v = 2.0 * (v - 0.5)

        v = v.float()
        f = f.long()

        v_encoded = self.sdf_grid_net.get_encoding(v)

        gaussians = self.gaussians_from_mesh(v_encoded.unsqueeze(0), v.unsqueeze(0), f.unsqueeze(0))

        if self.use_flexicubs:
            return gaussians, v, f, l
        else:
            return gaussians, v, f, n

class GaussianNetwork(nn.Module):
    def __init__(self, fc=None, fc_weights=None, resolution=128, device='cuda'):
        super(GaussianNetwork, self).__init__()

        self.device = device
        self.res = resolution
        self.multires = 6
        self.sdf_grid_net = ImplicitNetworkGrid(15, True, 3, 1, [64, 64], geometric_init=True, multires=self.multires, divide_factor=1.0).to(self.device)
        self.use_flexicubs = False
        if fc is not None:
            self.use_flexicubs = True
            self.mesh_from_grid = fc
            self.fc_weights = fc_weights
        else:
            self.mesh_from_grid = Grid2Mesh.apply
        self.gaussian_geo_from_mesh = GaussianGeoNetwork(D=64, num_geo_feat=16, num_hash_feat=32, num_sdf_feat=16, multires_geo=6).to(self.device)
        self.gaussian_rgb_from_mesh = GaussianRGBNetwork(D=64, num_geo_feat=16, num_hash_feat=32, multires_view=6).to(self.device)
        # self.gaussians_from_mesh = Mesh2GaussiansNetwork(multires=self.multires).to(self.device)

    def forward(self, input, cam_pose, cube_fx8=None):
        # Input is the regular grid with shape (res * res * res, 3)
        sdf_grid, _, grad_grid = self.sdf_grid_net.get_outputs(input)

        if self.use_flexicubs:
            v, f, l = self.mesh_from_grid(input, sdf_grid, cube_fx8, self.res, 
                                          self.fc_weights[:, :12], self.fc_weights[:, 12:20], self.fc_weights[:, 20], True, False, None)
        else:
            v, f, n = self.mesh_from_grid(sdf_grid.view(self.res, self.res, self.res), 0.0)
            v = 2.0 * (v - 0.5)

        v = v.float()
        f = f.long()

        v_encoded = self.sdf_grid_net.get_encoding(v) #(V, hash_feat + 3)
        v_sdf, v_feat, v_grad = self.sdf_grid_net.get_outputs(v)
        v_encoded = torch.cat([v_encoded, v_sdf, v_feat], dim=-1)

        # geo_feat : (G, geo_feat_num)
        # gaussian_xyz: (G, 3)
        gaussians, geo_feat = self.gaussian_geo_from_mesh(v_encoded.unsqueeze(0), v.unsqueeze(0), f.unsqueeze(0))

        gaussians_center_encoded = self.sdf_grid_net.get_encoding(gaussians['xyz']) # (G, hash_feat + 3)
        # sdf_gauss, _, grad_gauss = self.sdf_grid_net.get_outputs(gaussians['xyz'])

        grad = torch.cat([grad_grid, v_grad.squeeze(0)], dim=-2)
        
        view_dirs = F.normalize(cam_pose.unsqueeze(0) - gaussians['xyz']) # (G, 3)
        gaussians['rgb'] = self.gaussian_rgb_from_mesh(gaussians_center_encoded, geo_feat, view_dirs)

        ret = {
            'v' : v,
            'f' : f,
            'grad' : grad
        }

        if self.use_flexicubs:
            ret['l'] = l
            return gaussians, ret
        else:
            ret['n'] = n
            return gaussians, ret






    














        

