import os
import cv2
import sys
import uuid
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from random import randint

from gs.gs_renderer import render
from gs.gs_scene import Scene
from gs.gs_model import GaussianModel

from utils.loss_utils import l1_loss, ssim, sparse_loss
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.system_utils import load_config

# from scene.datasets import TestReplicaDataset, Replica, TUM_RGBD

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def eval_set(model_path, name, iteration, views, gaussians, pipe_cfg, background):
    render_path = os.path.join(model_path, name, "iter_{}".format(iteration), "renders")
    # gts_path = os.path.join(model_path, name, "iter_{}".format(iteration), "gt")

    os.makedirs(render_path, exist_ok=True)
    # os.makedirs(gts_path, exist_ok=True)

    l1_test = 0.0
    psnr_test = 0.0
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendereda_image = torch.clamp(render(view, gaussians, pipe_cfg, background)["image"], 0.0, 1.0)
        gt_image = torch.clamp(view.original_image.cuda(), 0.0, 1.0)
        # gt = view.original_image[0:3, :, :]
        l1_test += l1_loss(rendereda_image, gt_image).mean().double()
        psnr_test += psnr(rendereda_image, gt_image).mean().double()
        torchvision.utils.save_image(rendereda_image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
    l1_test /= len(views)
    psnr_test /= len(views)
    print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, name, l1_test.item(), psnr_test.item()))


def eval(scene, cfg_model, iteration, pipe_cfg, background, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        if not skip_train:
            eval_set(cfg_model['model_path'], "train", iteration, scene.getTrainCameras(), scene.gaussians, pipe_cfg, background)

        if not skip_test:
            eval_set(cfg_model['model_path'], "test", iteration, scene.getTestCameras(), scene.gaussians, pipe_cfg, background)

def training(cfg, scene : Scene, saving_iterations):
    cfg_training = cfg['training']
    cfg_model = cfg['model']
    
    first_iter = 0
    scene.gaussians.training_setup(cfg_training)

    bg_color = [1, 1, 1] if cfg['model']['white_background'] else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    iterations = cfg_training['iterations']
    progress_bar = tqdm(range(first_iter, iterations), desc="Training progress")
    first_iter += 1

    BCEloss = torch.nn.BCELoss()

    for iteration in range(first_iter, iterations):        

        iter_start.record()

        scene.gaussians.update_learning_rate(iteration)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        render_pkg = render(viewpoint_cam, scene.gaussians, cfg['pipeline'], background)
        image, depth, viewspace_point_tensor, visibility_filter, radii = render_pkg["image"], render_pkg["depth"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        loss = (1.0 - cfg_training['lambda_dssim']) * Ll1 + cfg_training['lambda_dssim'] * (1.0 - ssim(image, gt_image))
        
        # BCE loss for the opacity
        # if iteration >=  cfg_training['bce_from_iter'] and iteration < cfg_training['bce_from_iter'] + cfg_training['bce_iter_count']:
        #     loss += 0.0001 * sparse_loss(scene.gaussians.get_opacity[visibility_filter])
            # masked_opacity = scene.gaussians.get_opacity[visibility_filter]
            # loss += 0.05 * BCEloss(masked_opacity, masked_opacity)

        # Force the last scale to be small
        last_scale = scene.gaussians.get_scaling[visibility_filter][:, 2]
        loss += 0.01 * torch.mean(torch.abs(last_scale))
        
        
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == iterations:
                progress_bar.close()

            # Log and save
            if (iteration in saving_iterations):
                eval(scene, cfg_model, iteration, cfg['pipeline'], background, skip_train=False, skip_test=False)
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < cfg_training['densify_until_iter']:
                # Keep track of max radii in image-space for pruning
                scene.gaussians.max_radii2D[visibility_filter] = torch.max(scene.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                scene.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > cfg_training['densify_from_iter'] and iteration % cfg_training['densification_interval'] == 0:
                    size_threshold = 20 if iteration > cfg_training['opacity_reset_interval'] else None
                    scene.gaussians.densify_and_prune(cfg_training['densify_grad_threshold'], 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % cfg_training['opacity_reset_interval'] == 0 or (cfg_model['white_background'] and iteration == cfg_training['densify_from_iter']):
                    scene.gaussians.reset_opacity()

            # Optimizer step
            if iteration < iterations:
                scene.gaussians.optimizer.step()
                scene.gaussians.optimizer.zero_grad(set_to_none = True)

            # Extract coarse mesh and create new gaussians
            if (iteration == cfg_training['coarse_mesh_iter']):
                mesh_path = os.path.join(cfg_model['model_path'], 'mesh', 'iter{}_resol{}'.format(iteration, cfg_training['coarse_mesh_resolution']))
                scene.mesh = scene.gaussians.extract_mesh(mesh_path, density_thresh=0.1, resolution=cfg_training['coarse_mesh_resolution'])
                # scene.mesh.write_ply(mesh_path)
                scene.gaussians.create_from_mesh(scene.mesh)
                scene.gaussians.training_setup(cfg_training)

        torch.cuda.empty_cache()


if __name__ == "__main__":
    cfg = load_config('./configs/default.yaml')

    safe_state(False)
    torch.autograd.set_detect_anomaly(False)

    gaussians = GaussianModel(cfg['model']['sh_degree'])
    scene = Scene(cfg['model'], gaussians, load_iteration=None)

    training(cfg, scene, [i for i in range(0, 30000, 3000)])


