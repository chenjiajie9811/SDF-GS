import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable, List, Optional, Tuple, Union

from utils.system_utils import print_memory_usage

def near_far_from_sphere(rays_o, rays_d):
    a = torch.sum(rays_d**2, dim=-1, keepdim=True)

    b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
    mid = 0.5 * (-b) / a
    near = mid - 1.0
    far = mid + 1.0
    return near, far

def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples).cuda()
        u = u.expand(list(cdf.shape[:-1]) + [n_samples]).cuda()
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).cuda()

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1]).cuda() * torch.exp(self.variance * 10.0)

class NeuSRenderer(nn.Module):
    def __init__(
        self,
        sdf_network,
        deviation_network,
        num_samples,
        num_importance,
        upsample_steps,
        perturb=False
    ):
        super().__init__()
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.num_samples = num_samples
        self.num_importance = num_importance
        self.upsample_steps = upsample_steps
        self.perturb = perturb

    def upsample(self, rays_o, rays_d, z_vals, sdf, num_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        num_rays, num_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(num_rays, num_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros([num_rays, 1]).cuda(), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([num_rays, 1]).cuda(), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, num_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
        num_rays, num_samples = z_vals.shape
        _, num_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            new_sdf = self.sdf_network.get_sdf_vals(pts.reshape(-1, 3)).reshape(num_rays, num_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(num_rays)[:, None].expand(num_rays, num_samples + num_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(num_rays, num_samples + num_importance)

        return z_vals, sdf

    def render_depth(self, rays_o, rays_d, sample_dist, z_vals, sdf_network, deviation_network, cos_anneal_ratio=0.0):
        num_rays, num_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape).cuda()], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        sdf = sdf_network.get_sdf_vals(pts)#.reshape(num_rays, num_samples)
        gradients = sdf_network.gradient(pts)#.reshape(num_rays, num_samples, 3)

        inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
        inv_s = inv_s.expand(num_rays * num_samples, 1)

        true_cos = (dirs * gradients).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive
        
        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(num_rays, num_samples).clip(0.0, 1.0)

        weights = alpha * torch.cumprod(torch.cat([torch.ones([num_rays, 1]).cuda(), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        
        # print_memory_usage("depth rendering")

        depths = torch.sum(weights * z_vals, dim=-1)

        return depths, gradients


    def forward(self, rays_o, rays_d, near, far):
        # generate uniform samples
        num_rays = rays_o.shape[0]
        sample_dist = 2.0 / self.num_samples
        z_vals = torch.linspace(0.0, 1.0, self.num_samples).cuda()
        z_vals = near + (far - near) * z_vals[None, :] # (num_rays, num_samples)

        if self.perturb:
            t_rand = (torch.rand([num_rays, 1]) - 0.5).cuda()
            z_vals = z_vals + t_rand * 2.0 / self.num_samples

        num_samples = self.num_samples
        # upsample
        if self.num_importance > 0:
            with torch.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                sdf = self.sdf_network.get_sdf_vals(pts.reshape(-1, 3)).reshape(num_rays, self.num_samples)

                for i in range(self.upsample_steps):
                    new_z_vals = self.upsample(
                        rays_o, rays_d, z_vals, sdf, self.num_importance // self.upsample_steps, 64 * 2 **i)
                    
                    z_vals, sdf = self.cat_z_vals(
                        rays_o, rays_d, z_vals, new_z_vals, sdf, last=(i + 1 == self.upsample_steps))
                    
            num_samples = self.num_samples + self.num_importance


        return self.render_depth(rays_o, rays_d, sample_dist, z_vals, self.sdf_network, self.deviation_network)






        
    

