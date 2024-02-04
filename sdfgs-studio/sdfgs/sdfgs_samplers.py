import math
from abc import abstractmethod
from typing import Callable, List, Optional, Tuple, Union

import nerfacc
import torch
import torch.nn.functional as F
from nerfacc import OccGridEstimator
from torch import nn
from torchtyping import TensorType

from nerfstudio.cameras.rays import Frustums, RayBundle, RaySamples

from nerfstudio.model_components.ray_samplers import Sampler, UniformSampler, NeuSSampler

def s_density_func(inv_s, sdf):
    s = 1. / (inv_s)
    a = torch.exp(-s*sdf)
    b = a + 1
    return s * a / (b + 1) ** 2
 
class NeuSAccSampler(Sampler):
    def __init__(
        self, 
        aabb,
        neus_sampler: NeuSSampler = None,
        resolution: int = 128,
        num_samples: int = 8,
        num_samples_importance: int = 16,
        num_samples_boundary: int = 10,
        steps_warpup: int = 2000,
        steps_per_grid_update: int = 1000,
        importance_sampling: bool = False,
        local_rank: int = 0,
        single_jitter: bool = False,
    ):
        
        super().__init__()
        self.aabb = aabb
        self.resolution = resolution
        self.num_samples = num_samples
        self.num_samples_importance = num_samples_importance
        self.num_samples_boundary = num_samples_boundary
        self.single_jitter = single_jitter
        self.importance_sampling = importance_sampling
        self.steps_warpup = steps_warpup
        self.steps_per_grid_update = steps_per_grid_update
        self.local_rank = local_rank
        self.step_size = 0.01 / 5.0
        self.alpha_thres = 0.001

        # only supports cubic bbox for now
        assert aabb[0, 0] == aabb[0, 1] and aabb[0, 0] == aabb[0, 2]
        assert aabb[1, 0] == aabb[1, 1] and aabb[1, 0] == aabb[1, 2]
        self.grid_size = self.resolution
        self.voxel_size = (aabb[1, 0] - aabb[0, 0]) / self.grid_size

        # nesu_sampler at the begining of training
        # also use the pdf_sampler and outside_sampler in neus
        self.neus_sampler = neus_sampler

        self.grid = OccGridEstimator(aabb.reshape(-1), resolution=self.resolution, levels=4).train()
        self.register_buffer("_binary", torch.ones((self.grid_size, self.grid_size, self.grid_size), dtype=torch.bool))
        self.register_buffer("_update_counter", torch.zeros(1, dtype=torch.int32))

        self.init_grid_coordinate()

    def init_grid_coordinate(self):
        # coarse grid coordinates
        aabb = self.aabb
        offset_x = torch.linspace(
            aabb[0, 0] + self.voxel_size / 2.0, aabb[1, 0] - self.voxel_size / 2.0, self.grid_size
        )
        offset_y = torch.linspace(
            aabb[0, 1] + self.voxel_size / 2.0, aabb[1, 1] - self.voxel_size / 2.0, self.grid_size
        )
        offset_z = torch.linspace(
            aabb[0, 2] + self.voxel_size / 2.0, aabb[1, 2] - self.voxel_size / 2.0, self.grid_size
        )
        x, y, z = torch.meshgrid(offset_x, offset_y, offset_z, indexing="ij")
        cube_coordinate = torch.stack([x, y, z], dim=-1).reshape(-1, 3)

        self.register_buffer("cube_coordinate", cube_coordinate)

    def update_step_size(self, step, inv_s=None):
        assert inv_s is not None
        inv_s = inv_s().item()
        self.step_size = 14.0 / inv_s / 16

    @torch.no_grad()
    def update_binary_grid(self, step, sdf_fn=None, inv_s=None):
        assert sdf_fn is not None
        assert inv_s is not None

        if step >= self.steps_warpup and step % self.steps_per_grid_update == 0:
            mask = self._binary.reshape(-1)
            occupied_voxel = self.cube_coordinate[mask.reshape(-1)]

            def evaluate(points):
                z = []
                for _, pnts in enumerate(torch.split(points, 100000, dim=0)):
                    z.append(sdf_fn(pnts))
                z = torch.cat(z, axis=0)
                return z
            
            sdf = evaluate(occupied_voxel)

            # use maximum bound for sdf value
            bound = self.voxel_size * (3**0.5) / 2.0
            sdf = sdf.abs()
            sdf = torch.maximum(sdf - bound, torch.zeros_like(sdf))

            estimated_next_sdf = sdf - self.step_size * 0.5
            estimated_prev_sdf = sdf + self.step_size * 0.5
            inv_s = inv_s()
            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

            p = prev_cdf - next_cdf
            c = prev_cdf

            alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)
            sdf_mask = alpha > self.alpha_thres

            mask[mask.reshape(-1).clone()] = sdf_mask

            self._binary = mask.reshape([self.grid_size] * 3).contiguous()

            # save_points("voxel_valid.ply", self.cube_coordinate[self._binary.reshape(-1)].cpu().numpy())

            # TODO do we need dilation
            # F.max_pool3d(M.float(), kernel_size=3, padding=1, stride=1).bool()
            # save_points("voxel_valid_dilated.ply", self.cube_coordinate[self._binary.reshape(-1)].cpu().numpy())
            self._update_counter += 1

    def create_ray_samples_from_ray_indices(self, ray_bundle: RayBundle, ray_indices, t_starts, t_ends):
        rays_o = ray_bundle.origins[ray_indices]
        rays_d = ray_bundle.directions[ray_indices]
        camera_indices = ray_bundle.camera_indices[ray_indices]
        deltas = t_ends - t_starts

        frustums = Frustums(
            origins=rays_o,  # [..., 1, 3]
            directions=rays_d,  # [..., 1, 3]
            starts=t_starts,  # [..., num_samples, 1]
            ends=t_ends,  # [..., num_samples, 1]
            pixel_area=torch.ones_like(t_starts),  # [..., 1, 1]
        )

        ray_samples = RaySamples(
            frustums=frustums,
            camera_indices=camera_indices,  # [..., 1, 1]
            deltas=deltas,  # [..., num_samples, 1]
        )
        return ray_samples

    @torch.no_grad()
    def generate_ray_samples(
        self,
        ray_bundle: Optional[RayBundle] = None,
        sdf_fn: Optional[Callable] = None,
        alpha_fn: Optional[Callable] = None,
        neus_fields = None
    ) -> Union[Tuple[RaySamples, torch.Tensor], RaySamples]:
        assert ray_bundle is not None
        assert sdf_fn is not None

        # sampler with original neus Sampler?
        # if self._update_counter.item() <= 0:
        #     return self.neus_sampler(ray_bundle, sdf_fn=sdf_fn)

        assert alpha_fn is not None

        #@TODO check https://github.com/nerfstudio-project/nerfacc/blob/master/examples/utils.py#L54
        # to modify the sampling method 
        def occ_eval_fn(x):
            hidden_output = neus_fields.forward_geonetwork(x)
            sdf, _ = torch.split(hidden_output, [1, neus_fields.config.geo_feat_dim], dim=-1)
            return -sdf * 1e-3
        
        self.grid._update(
            step=self._update_counter.item(),
            occ_eval_fn=occ_eval_fn,
            occ_thre= 0.01
        )

        def alpha_fn_(t_starts, t_ends, ray_indices):
            t_origins = ray_bundle.origins[ray_indices]
            t_dirs = ray_bundle.directions[ray_indices]
            inputs = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0

            hidden_output = neus_fields.forward_geonetwork(inputs)
            sdf, _ = torch.split(hidden_output, [1, neus_fields.config.geo_feat_dim], dim=-1)
            return (s_density_func(neus_fields.deviation_network.get_variance(), sdf)).squeeze(-1)

        # sampling from occupancy grids
        ray_indices, t_starts, t_ends = self.grid.sampling(
            rays_o=ray_bundle.origins.contiguous(),
            rays_d=ray_bundle.directions.contiguous(),
            alpha_fn=alpha_fn_,
            # near_plane=ray_bundle.nears[:, 0].contiguous(),
            # far_plane=ray_bundle.fars[:, 0].contiguous(),
            alpha_thre=0.01
        )

        # print (torch.unique(ray_indices))
        # print (ray_indices.shape)
        # input()

        # create ray_samples with the intersection
        ray_indices = ray_indices.long()
        ray_samples = self.create_ray_samples_from_ray_indices(ray_bundle, ray_indices, t_starts[...,None], t_ends[...,None])

        # total_iters = 0
        # sorted_index = None
        # sdf: Optional[torch.Tensor] = None
        # new_samples = ray_samples
        # print ("line 208 ",new_samples.shape)

        # base_variance = self.neus_sampler.base_variance

        # while total_iters < self.neus_sampler.num_upsample_steps:
        #     with torch.no_grad():
        #         new_sdf = sdf_fn(new_samples)

        #     # merge sdf predictions
        #     if sorted_index is not None:
        #         assert sdf is not None
        #         sdf_merge = torch.cat([sdf.squeeze(-1), new_sdf.squeeze(-1)], -1)
        #         sdf = torch.gather(sdf_merge, 1, sorted_index)#.unsqueeze(-1)
        #     else:
        #         sdf = new_sdf

        #     # compute with fix variances
        #     alphas = self.neus_sampler.rendering_sdf_with_fixed_inv_s(
        #         ray_samples, sdf.reshape(ray_samples.shape, 1), inv_s=base_variance * 2**total_iters
        #     )

        #     weights = nerfacc.render_weight_from_alpha(alphas=alphas, ray_indices=ray_indices, n_rays=ray_bundle.shape[0])
        #     weights = torch.cat((weights, torch.zeros_like(weights[:, :1])), dim=1)

        #     new_samples = self.neus_sampler.pdf_sampler(
        #         ray_bundle,
        #         ray_samples,
        #         weights,
        #         num_samples=self.num_samples_importance // self.neus_sampler.num_upsample_steps,
        #     )

        #     ray_samples, sorted_index = self.neus_sampler.merge_ray_samples(ray_bundle, ray_samples, new_samples)
        #     total_iters += 1



        # if self.importance_sampling and ray_samples.shape[0] > 0:
        #     # save_points("first.ply", ray_samples.frustums.get_start_positions().cpu().numpy().reshape(-1, 3))

        #     alphas = alpha_fn(ray_samples)
        #     weights = nerfacc.render_weight_from_alpha(alphas, ray_indices=ray_indices, n_rays=ray_bundle.shape[0])

        #     # TODO make it configurable
        #     # re sample
        #     packed_info, t_starts, t_ends = nerfacc.ray_resampling(packed_info, t_starts, t_ends, weights[:, 0], 16)
        #     ray_indices = nerfacc.unpack_info(packed_info, t_starts.shape[0])
        #     ray_samples = self.create_ray_samples_from_ray_indices(ray_bundle, ray_indices, t_starts, t_ends)

            # save_points("second.ply", ray_samples.frustums.get_start_positions().cpu().numpy().reshape(-1, 3))
        self._update_counter += 1
        return ray_samples, ray_indices