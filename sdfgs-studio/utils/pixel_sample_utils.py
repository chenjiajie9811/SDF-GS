import torch
import random
import numpy as np

from skimage.color import rgb2gray
from skimage import filters

def get_rays_from_uv(i, j, c2w, fx, fy, cx, cy, device='cuda:0'):
    """
    Get corresponding rays from input uv.
    i,j are flattened.

    """
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w).to(device)

    dirs = torch.stack(
        [(i-cx)/fx, -(j-cy)/fy, torch.ones_like(i)], 1).to(device)
    dirs = dirs.reshape(-1, 1, 3)

    rays_d = torch.sum(dirs * c2w[:3, :3], -1)

    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d

def get_sample_indices_with_grad(n, image, mask=None):
    """
    Sample n uv coordinates from an image region H0..H1, W0..W1 with the intensity gradient
    mask: foreground

    """
    intensity = rgb2gray(image.cpu().numpy())
    grad_y = filters.sobel_h(intensity)
    grad_x = filters.sobel_v(intensity)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    if mask is not None:
        grad_mag[~mask] = 0

    img_size = (image.shape[0], image.shape[1])
    selected_index = np.argpartition(grad_mag, -5*n, axis=None)[-5*n:]
    indices_h, indices_w = np.unravel_index(selected_index, img_size)
    # selected_index = np.ravel_multi_index(
    #     np.array((indices_h, indices_w)), img_size)

    samples = np.random.choice(
        range(0, indices_h.shape[0]), size=n, replace=False)
    indices_h = indices_h[samples].reshape(-1, 1)
    indices_w = indices_w[samples].reshape(-1, 1)
    indices = torch.from_numpy(np.concatenate([indices_h, indices_w], axis=1))

    return indices

def get_sample_indices_random(n, image, mask=None, device='cuda:0'):

    """
    mask: foreground
    """
    
    if mask is not None:
        nonezero_indices = torch.nonzero(mask)
        chosen_indices = random.sample(range(len(nonezero_indices)), k=n)
        indices = nonezero_indices[chosen_indices]

    else:
        indices = torch.floor(
            torch.rand((n, 2), device=device)
            * torch.tensor([image.shape[0], image.shape[1]], device=device)
        ).long()

    return indices

def get_samples_with_pixel_grad_and_random(
                                           n_grad, n_rand,
                                           fx, fy, cx, cy, c2w, color, mask=None,
                                            device='cuda:0'):
    
    index_grad = torch.empty(0)
    index_rand = torch.empty(0)

    color = color.permute(1, 2, 0)
    
    if n_grad > 0:
        index_grad = get_sample_indices_with_grad(n_grad, color, mask)
        
    if n_rand > 0:
        index_rand = get_sample_indices_random(n_rand, color, mask)

    indices = torch.unique(torch.cat([index_grad, index_rand], dim=0), dim=0)

    # print (indices.shape)

    # import matplotlib.pyplot as plt

    # plt.imshow(color.cpu().numpy())
    # plt.scatter(indices[:, 1], indices[:, 0], s=1, c='red')
    # plt.savefig("/usr/stud/chj/storage/user/chj/SDF-GS/samples.png")
    sampled_color = color[indices[:, 0], indices[:, 1], :]

    rays_o, rays_d = get_rays_from_uv(indices[:, 1], indices[:, 0], c2w, fx, fy, cx, cy, device)
    
    return rays_o, rays_d, indices, sampled_color

def get_samples_random(n_samples, fx, fy, cx, cy, c2w, color, mask=None, device='cuda:0'):
    color = color.permute(1, 2, 0)
    indices = get_sample_indices_random(n_samples, color, mask)
    sampled_color = color[indices[:, 0], indices[:, 1], :]
    rays_o, rays_d = get_rays_from_uv(indices[:, 1], indices[:, 0], c2w, fx, fy, cx, cy, device)

    # import matplotlib.pyplot as plt

    # dummy = torch.zeros_like(color).cpu().numpy()
    # dummy[indices[:, 0].cpu().numpy(), indices[:, 1].cpu().numpy(), :] = sampled_color.cpu().numpy()
    # plt.imshow(dummy)
    # plt.savefig("/usr/stud/chj/storage/user/chj/SDF-GS/samples.png")
    # print ("saved!")
    
    return rays_o, rays_d, indices, sampled_color


def get_all_rays(H, W, fx, fy, cx, cy, c2w, device='cuda', crop_edge=0, down_scale=1):
    """
    Get rays for a whole image.

    """
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w)

    H_step, W_step = H-2*crop_edge // down_scale, W-2*crop_edge // down_scale
    i, j = torch.meshgrid(torch.linspace(crop_edge, W-1-crop_edge, W_step),
                          torch.linspace(crop_edge, H-1-crop_edge, H_step), indexing='ij')
    i = i.t().long()
    j = j.t().long()

    dirs = torch.stack(
        [(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1).to(device)
    dirs = dirs.reshape(H_step, W_step, 1, 3)
    rays_d = torch.sum(dirs * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)

    return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)










    








