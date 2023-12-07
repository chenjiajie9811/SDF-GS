import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from grid.embedder import *
from utils.rigid_utils import exp_so3
from grid.grid import Grid2Mesh, DenseGrid

class Mesh2GaussiansNetwork(nn.Module):
    def __init__(self, L=4, D=64, multires=10):
        super(Mesh2GaussiansNetwork, self).__init__()
        self.L = L
        self.D = D
        self.skips = [L // 2]

        self.embed_fn, vertex_input_ch = get_embedder(multires, 3)

        self.input_ch = 3 * vertex_input_ch + 3 # (3 embedded vertices + 3 normals)

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

    def forward(self, vertices, faces, normals):
        vertices_emb = self.embed_fn(vertices) # (B, M, K)
        print (vertices_emb)

        vec1 = vertices[:, faces[:, :, 0].flatten()] - vertices[:, faces[:, :, 1].flatten()] 
        vec2 = vertices[:, faces[:, :, 0].flatten()] - vertices[:, faces[:, :, 2].flatten()] 
        faces_normal = F.normalize(torch.cross(vec1, vec2, dim=-1))
        # faces_normal = F.normalize((normals[:, faces[:, :, 0].flatten()] + normals[:, faces[:, :, 1].flatten()] + normals[:, faces[:, :, 2].flatten()]) / 3.0, dim=-1) 
        x = torch.cat([vertices_emb[:, faces[:, :, 0].flatten()], vertices_emb[:, faces[:, :, 1].flatten()], vertices_emb[:, faces[:, :, 2].flatten()], faces_normal], dim=-1) 
        h = self.linear[0](x)
        h = F.relu(h)
        for i in range(1, self.D - 1):
            h = self.linear[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x, h], dim=-1)

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
    
        print (ret['rgb'])
        print (ret['opacity'])
        print (ret['scaling'])
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

        return ret












        

