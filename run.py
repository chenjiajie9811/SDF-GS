import torch
import torchvision
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation
from dpsr import *
# from grid.grid import *

from tqdm import tqdm
from random import randint

from gs.gs_renderer import render
from gs.gs_scene import Scene
from gs.gs_model import GaussianModel

from utils.loss_utils import l1_loss, ssim, sparse_loss
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.system_utils import load_config

from train import eval

def read_pc_from_ply(path):
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
    scales = np.exp(scales)

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names))) # (w, x, y, z)
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rots_ = np.zeros_like(rots)
    rots_[:, 3] = rots[:, 0]
    rots_[:, :3] = rots[:, 1:] 
    r = Rotation.from_quat(rots_)

    rotation_matrices = r.as_matrix()
    normals = np.zeros((xyz.shape[0], 3))


    axis_id = np.argmin(scales, axis=1)
    axes = np.zeros((xyz.shape[0], 3))
    axes[np.arange(xyz.shape[0]), axis_id.flatten()] = 1
    normals = np.einsum('ijk, ik -> ij', rotation_matrices, axes)

    centers = torch.tensor(xyz, dtype=torch.float, device="cpu")
    normals = torch.tensor(normals, dtype=torch.float, device="cpu")
    
    return centers, normals

def save_points_ply(points, path):
    vertex = np.array([(points[i][0], points[i][1], points[i][2]) 
                        for i in range(points.shape[0])], 
                        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    
    vertex_element = PlyElement.describe(vertex, 'vertex')
    plydata = PlyData([vertex_element])
    plydata.write(path)

def normalize_coordinate(points):
    min_cor = torch.min(points, dim=0)
    print (min_cor)
    max_cor = torch.max(points, dim=0)

    #max_cor[max_cor == min_cor] = 1.0

    return (points - min_cor.values) / (max_cor.values.max())

def training(cfg, scene: Scene, saving_iterations):
    cfg_training = cfg['training']
    cfg_model = cfg['model']
    
    first_iter = 0
    scene.gaussians.training_setup_second_stage(cfg_training)

    bg_color = [1, 1, 1] if cfg['model']['white_background'] else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    iterations = cfg_training['iterations']
    progress_bar = tqdm(range(first_iter, iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, iterations):        

        iter_start.record()

        scene.gaussians.update_learning_rate(iteration)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Update one iteration
        scene.gaussians.update_second_stage()
        
        # Render
        render_pkg = render(viewpoint_cam, scene.gaussians, cfg['pipeline'], background)
        image, depth, viewspace_point_tensor, visibility_filter, radii = render_pkg["image"], render_pkg["depth"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        loss = (1.0 - cfg_training['lambda_dssim']) * Ll1 + cfg_training['lambda_dssim'] * (1.0 - ssim(image, gt_image))
        
        loss.backward(retain_graph=True)

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

            # Optimizer step
            # if iteration < iterations:
            scene.gaussians.optimizer.step()
            scene.gaussians.optimizer.zero_grad(set_to_none = True)

        # torch.cuda.empty_cache()

    



def demo_training_stage_2():
    cfg = load_config('./configs/default.yaml')

    safe_state(False)
    torch.autograd.set_detect_anomaly(False)

    gaussian_path = './output/lego/point_cloud/iteration_6000/point_cloud.ply'
    mesh_path = './output/mesh_128_from_iter3000.ply'

    gaussians = GaussianModel(2)
    gaussians.load_ply(gaussian_path)
    gaussians.init_second_stage(debug_save=False, debug_load=True)
    scene = Scene(cfg['model'], gaussians, load_iteration=None)

    training(cfg, scene, [i for i in range(0, 30000, 30)])

if __name__ == '__main__':
    # import cv2
    # a = cv2.imread('/storage/user/chj/chj/SDF-GS/output/lego/test/iter_3000/renders/00000.png')
    # print (a.max())
    demo_training_stage_2()
    
    # V, N = read_pc_from_ply('/usr/stud/chj/storage/user/chj/SDF-GS/output/lego/point_cloud/iteration_9000/point_cloud.ply')
    # dpsr = DPSR((64, 64, 64)).cuda()
    # V = normalize_coordinate(V)
    # print (V)
    # V = V.unsqueeze(0)
    # N = N.unsqueeze(0)
    # print(V.shape)
    # grid = dpsr(V, N)

    # v, f, n = mc_from_psr(grid)
    # print(v.shape)
    # print (f.shape)
    # print (n.shape)

    gaussian_path = './output/lego/point_cloud/iteration_6000/point_cloud.ply'
    mesh_path = './output/mesh_128_from_iter3000.ply'

    gaussians = GaussianModel(2)
    gaussians.load_ply(gaussian_path)

    gaussians.init_second_stage(debug_save=False, debug_load=True)
    #gaussians.training_setup_second_stage()
    gaussians.update_second_stage()

    # save_points_ply(gaussians.get_xyz, './sampled.ply')
    
    # # from grid.grid import DenseGrid
    
    # mesh = gaussians.extract_mesh(mesh_path, density_thresh=0.1, resolution=128)
    # mesh.write_ply(mesh_path)



    # import trimesh
    # data = trimesh.load(mesh_path)

    # pts = V
    # mn, mx = pts.amin(0), pts.amax(0)
    # grid_center = (mn + mx) / 2
    # grid_scale = 1. / (mx - mn).amax().item() 

    # pts = (pts - grid_center) * grid_scale / 2
    # pts = pts + torch.tensor([0.5, 0.5, 0.5])

    # print(pts.amin(0), " " , pts.amax(0))
    # # for p in pts:
    # #     print (p)
    # # input()

    # vals = N
    # pts = pts.cuda()
    # vals = vals.cuda()

    # grid = dpsr(pts.unsqueeze(0), vals.unsqueeze(0))
    # v, f, n = mc_from_psr(grid)
    # print(v.shape)
    # print (f.shape)
    # print (n.shape)

    # save_points_ply(v, './sampled.ply')
    # raster = point_rasterize(pts.unsqueeze(0), vals.unsqueeze(0), [64, 64, 64])
    # print (raster)


    # bary_weights = generate_bary_coords_params(data.triangles.shape[0])
    # centers = sample_surface_points_with_barycentric(torch.tensor(data.vertices), torch.tensor(data.faces), None)

    # from utils.mesh import Mesh
    # from utils.mesh_utils import sample_points_with_barycentric
    # mesh = Mesh(v=torch.tensor(data.vertices), f=torch.tensor(data.faces))
    
    # sampled_points = sample_points_with_barycentric(mesh)
    # save_points_ply(centers, './sampled.ply')
    # print (centers.shape)

    # from simple_knn._C import distCUDA2
    # dist2 = distCUDA2(gaussians.get_xyz)
    # print (dist2.shape)

    






