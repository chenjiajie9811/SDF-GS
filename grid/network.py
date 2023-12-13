import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from grid.embedder import *
from utils.rigid_utils import exp_so3
from grid.grid import Grid2Mesh, DenseGrid


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
            base_size = 8, #16,
            end_size = 1024,#2048,
            logmap = 16,#19,
            num_levels=8,#16,
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
    def __init__(self, L=4, D=128, multires=2):
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
        faces_normal = F.normalize(torch.cross(vec1, vec2, dim=-1))
        x = torch.cat([embeded[:, faces[:, :, 0].flatten()], embeded[:, faces[:, :, 1].flatten()], embeded[:, faces[:, :, 2].flatten()], faces_normal], dim=-1) 
        
        h = self.linear[0](x)

        h = F.relu(h)
        for i in range(1, self.L):
            h = self.linear[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x, h], dim=-1)
            # print ("output at layer: ", i)
            # print (h.shape)
        
        bary_coords_2 = torch.sigmoid(self.bary_coords_lin(h))
        bary_coord_last = 1. - torch.sum(bary_coords_2, dim=-1)

        ret = dict()

        ret['xyz'] =  vertices[:, faces[:, :, 0].flatten()] * bary_coords_2[:, :, 0].reshape(-1, 1).unsqueeze(0) + \
                                    vertices[:, faces[:, :, 0].flatten()] * bary_coords_2[:, :, 1].reshape(-1, 1).unsqueeze(0) + \
                                    vertices[:, faces[:, :, 0].flatten()] * bary_coord_last.reshape(-1, 1).unsqueeze(0)
        
        ret['opacity'] = torch.sigmoid(self.opacity_lin(h))

        ret['scaling']= torch.sigmoid(self.scaling_lin(h)) # need the base scaling
        # print (ret['scaling'].shape)
        # print (torch.sigmoid(self.scaling_lin(h)).shape)
        # print (ret['scaling'])
        # print (torch.sigmoid(self.scaling_lin(h)))
        rotation_theta = torch.sigmoid(self.rotation_angle_lin(h)) * 2 * torch.pi
        
        
        ret['rotation_matrix'] = exp_so3(faces_normal.squeeze(0), rotation_theta.squeeze(0)).unsqueeze(0)
        # ret['rotation'] = F.normalize(self.rotation_quat_lin(h))
        ret['rgb'] = torch.sigmoid(self.rgb_lin(h))

        return ret
    
    
    # def forward(self, vertices, faces, normals):
    #     vertices_emb = self.embed_fn(vertices) # (B, M, K)
    #     # print (vertices_emb)

    #     vec1 = vertices[:, faces[:, :, 0].flatten()] - vertices[:, faces[:, :, 1].flatten()] 
    #     vec2 = vertices[:, faces[:, :, 0].flatten()] - vertices[:, faces[:, :, 2].flatten()] 
    #     faces_normal = F.normalize(torch.cross(vec1, vec2, dim=-1))
    #     # faces_normal = F.normalize((normals[:, faces[:, :, 0].flatten()] + normals[:, faces[:, :, 1].flatten()] + normals[:, faces[:, :, 2].flatten()]) / 3.0, dim=-1) 
    #     x = torch.cat([vertices_emb[:, faces[:, :, 0].flatten()], vertices_emb[:, faces[:, :, 1].flatten()], vertices_emb[:, faces[:, :, 2].flatten()], faces_normal], dim=-1) 
        
    #     h = self.linear[0](x)

    #     h = F.relu(h)
    #     for i in range(1, self.L):
    #         h = self.linear[i](h)
    #         h = F.relu(h)
    #         if i in self.skips:
    #             h = torch.cat([x, h], dim=-1)
    #         # print ("output at layer: ", i)
    #         # print (h.shape)
        
    #     bary_coords_2 = torch.sigmoid(self.bary_coords_lin(h))
    #     bary_coord_last = 1. - torch.sum(bary_coords_2, dim=-1)

    #     ret = dict()

    #     ret['xyz'] =  vertices[:, faces[:, :, 0].flatten()] * bary_coords_2[:, :, 0].reshape(-1, 1).unsqueeze(0) + \
    #                                 vertices[:, faces[:, :, 0].flatten()] * bary_coords_2[:, :, 1].reshape(-1, 1).unsqueeze(0) + \
    #                                 vertices[:, faces[:, :, 0].flatten()] * bary_coord_last.reshape(-1, 1).unsqueeze(0)
        
    #     ret['opacity'] = torch.sigmoid(self.opacity_lin(h))

    #     ret['scaling']= torch.sigmoid(self.scaling_lin(h)) # need the base scaling
    #     # print (ret['scaling'].shape)
    #     # print (torch.sigmoid(self.scaling_lin(h)).shape)
    #     # print (ret['scaling'])
    #     # print (torch.sigmoid(self.scaling_lin(h)))
    #     rotation_theta = torch.sigmoid(self.rotation_angle_lin(h)) * 2 * torch.pi
    #     ret['rotation_matrix'] = exp_so3(faces_normal.squeeze(0), rotation_theta.squeeze(0)).unsqueeze(0)
    #     # ret['rotation'] = F.normalize(self.rotation_quat_lin(h))
    #     ret['rgb'] = torch.sigmoid(self.rgb_lin(h))
    
        # print (ret['rgb'])
        # print (ret['opacity'])
        # print (ret['scaling'])
        # ret = {
        #     'xyz' : torch.zeros_like(faces),
        #     'opacity' : torch.zeros((*faces.shape[:2], 1)),
        #     'scaling' : torch.zeros_like(faces),
        #     'rotation' : torch.zeros((*faces.shape[:2], 4)),
        #     'rgb' : torch.zeros_like(faces)
        # }
        
        # batch_size = 100000
        # for start in range(0, faces.shape[1], batch_size):
        #     end = min(start + batch_size, faces.shape[1])
        
        #     faces_normal = F.normalize((normals[:, faces[:, start : end, 0].flatten()] + normals[:, faces[:, start : end, 1].flatten()] + normals[:, faces[:, start : end, 2].flatten()]) / 3.0, dim=-1) 
        #     x = torch.cat([vertices_emb[:, faces[:, start : end, 0].flatten()], vertices_emb[:, faces[:, start : end, 1].flatten()], vertices_emb[:, faces[:, start : end, 2].flatten()], faces_normal], dim=-1) 
        #     h = self.linear[0](x)
        #     h = F.relu(h)
        #     for i in range(1, self.D - 1):
        #         h = self.linear[i](h)
        #         h = F.relu(h)
        #         if i in self.skips:
        #             h = torch.cat([x, h], dim=-1)

        #     bary_coords_2 = torch.sigmoid(self.bary_coords_lin(h))
        #     bary_coord_last = 1. - torch.sum(bary_coords_2, dim=-1)

        #     ret['xyz'][:, start:end] =  vertices[:, faces[:, start : end, 0].flatten()] * bary_coords_2[:, :, 0].reshape(-1, 1).unsqueeze(0) + \
        #                                 vertices[:, faces[:, start : end, 0].flatten()] * bary_coords_2[:, :, 1].reshape(-1, 1).unsqueeze(0) + \
        #                                 vertices[:, faces[:, start : end, 0].flatten()] * bary_coord_last.reshape(-1, 1).unsqueeze(0)
            
        #     ret['opacity'][:, start:end] = torch.sigmoid(self.opacity_lin(h))

        #     ret['scaling'][:, start:end] = torch.sigmoid(self.scaling_lin(h)) # need the base scaling
        #     print (ret['scaling'][:, start:end].shape)
        #     print (torch.sigmoid(self.scaling_lin(h)).shape)
        #     print (ret['scaling'][:, start:end])
        #     print (torch.sigmoid(self.scaling_lin(h)))
        #     # rotation_theta = torch.sigmoid(self.rotation_angle_lin(h)) * 2 * torch.pi
        #     # ret['rotation_matrix'] = exp_so3(faces_normal, rotation_theta)
        #     ret['rotation'][:, start:end] = F.normalize(self.rotation_quat_lin(h))
        #     ret['rgb'][:, start:end] = torch.sigmoid(self.rgb_lin(h))

        # # print (ret)

        # return ret
    

class GaussianNetwork(nn.Module):
    def __init__(self, device='cuda'):
        super(GaussianNetwork, self).__init__()

        self.device = device

        self.sdf_grid_net = ImplicitNetworkGrid(1, True, 3, 1, [64, 64], multires=2, divide_factor=1.0).to(self.device)
        self.mesh_from_grid = Grid2Mesh.apply
        self.gaussians_from_mesh = Mesh2GaussiansNetwork().to(self.device)

    def forward(self, input):
        # Input is the regular grid with shape (res * res * res, 3)
        grid = self.sdf_grid_net(input)
        sdf_grid = grid[:, :1]
        features_grid = grid[:, 1:]

        v, f, n = self.mesh_from_grid(sdf_grid.view(64, 64, 64), 0.0)
        v = 2.0 * (v - 0.5)
        v = v.float()
        f = f.long()

        v_encoded = self.sdf_grid_net.get_encoding(v)

        gaussians = self.gaussians_from_mesh(v_encoded.unsqueeze(0), v.unsqueeze(0), f.unsqueeze(0))

        return gaussians






    














        

