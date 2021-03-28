import os
import argparse
import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from utils import utils, warp, image_utils, constant_var
from models import end_2_end_optimization
from options import fake_options
import cv2

import sys
import pdb
import pickle

sys.path.insert(1, '/home/chrizandr')

from yolov3.annotations.annot.annot_utils import CVAT_Track, CVAT_annotation


# if want to run on CPU, please make it False
constant_var.USE_CUDA = True
utils.fix_randomness()
# if GPU is RTX 20XX, please disable cudnn
torch.backends.cudnn.enabled = True


def get_opt():
    opt = fake_options.FakeOptions()
    opt.batch_size = 1
    opt.coord_conv_template = True
    opt.error_model = 'loss_surface'
    opt.error_target = 'iou_whole'
    opt.goal_image_path = './data/world_cup_2018.png'
    opt.guess_model = 'init_guess'
    opt.homo_param_method = 'deep_homography'
    opt.load_weights_error_model = 'pretrained_loss_surface'
    opt.load_weights_upstream = 'pretrained_init_guess'
    opt.lr_optim = 1e-5
    opt.need_single_image_normalization = True
    opt.need_spectral_norm_error_model = True
    opt.need_spectral_norm_upstream = False
    opt.optim_criterion = 'l1loss'
    opt.optim_iters = 200
    opt.optim_method = 'stn'
    opt.optim_type = 'adam'
    opt.out_dir = './out'
    opt.prevent_neg = 'sigmoid'
    opt.template_path = './data/world_cup_template.png'
    opt.warp_dim = 8
    opt.warp_type = 'homography'
    return opt


def read_img(img_name, opt):
    goal_image = imageio.imread(img_name, pilmode='RGB')
    pil_image = Image.fromarray(np.uint8(goal_image))
    pil_image = pil_image.resize([256, 256], resample=Image.NEAREST)
    goal_image = np.array(pil_image)
    goal_image = utils.np_img_to_torch_img(goal_image)
    if opt.need_single_image_normalization:
        goal_image = image_utils.normalize_single_image(goal_image)
    return goal_image


def read_template(opt):
    template_image = imageio.imread(opt.template_path, pilmode='RGB')
    template_image = template_image / 255.0
    if opt.coord_conv_template:
        template_image = image_utils.rgb_template_to_coord_conv_template(template_image)
    template_image = utils.np_img_to_torch_img(template_image)
    if opt.need_single_image_normalization:
        template_image = image_utils.normalize_single_image(template_image)
    return template_image


def warp_point(x, y, optim_homography):
    frame_point = np.array([x, y])

    x = torch.tensor(frame_point[0] / 1280 - 0.5).float()
    y = torch.tensor(frame_point[1] / 720 - 0.5).float()
    xy = torch.stack([x, y, torch.ones_like(x)])
    xy_warped = torch.matmul(optim_homography.cpu(), xy)  # H.bmm(xy)
    xy_warped, z_warped = xy_warped.split(2, dim=1)

    # we multiply by 2, since our homographies map to
    # coordinates in the range [-0.5, 0.5] (the ones in our GT datasets)
    xy_warped = 2.0 * xy_warped / (z_warped + 1e-8)
    x_warped, y_warped = torch.unbind(xy_warped, dim=1)
    # [-1, 1] -> [0, 1]
    x_warped = (x_warped.item() * 0.5 + 0.5) * 1050
    y_warped = (y_warped.item() * 0.5 + 0.5) * 680
    return x_warped, y_warped


def run(data_path, opt, annotation, savefile):
    e2e = end_2_end_optimization.End2EndOptimFactory.get_end_2_end_optimization_model(opt)
    template_image = read_template(opt)
    heatmap = np.zeros(template_image.shape[1::], dtype=np.float64)
    imgs = os.listdir(data_path)
    total_imgs = len(imgs)
    frames = [int(x.strip("image").strip(".jpg")) for x in imgs]
    frames.sort()
    assert frames == [x for x in range(1, total_imgs+1)]
    sampled_imgs = [x for x in range(1, total_imgs+1, 60)]
    prefix = "image%03d"
    annot = CVAT_annotation(annotation)
    annot.index_by_frame()

    for frame in tqdm(sampled_imgs):
        if frame-1 not in annot.frames:
            continue
        img_name = os.path.join(data_path, prefix % frame + ".jpg")
        goal_image = read_img(img_name, opt)
        orig_homography, optim_homography = e2e.optim(goal_image[None], template_image)
        bboxes = annot.frames[frame - 1]

        for bbox in bboxes:
            cx, cy = (bbox.xtl + bbox.xbr) / 2, (bbox.ytl + bbox.ybr) / 2
            x_warped, y_warped = warp_point(cx, cy, optim_homography)
            x_warped, y_warped = int(x_warped), int(y_warped)
            if x_warped < heatmap.shape[1] and y_warped < heatmap.shape[0] and x_warped >= 0 and y_warped >= 0:
                heatmap[int(y_warped), int(x_warped)] += 1
    pickle.dump(heatmap, open(savefile, "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--annot', default="/home/chrizandr/annot/frvsbe.xml", type=str, help='CVAT Annotation file')
    parser.add_argument('--data', default="/ssd_scratch/cvit/chrizandr/images", type=str, help='Path where the generated labels are saved')
    parser.add_argument('--savefile', default="heatmap.pkl", type=str, help='Path where the generated labels are saved')
    args = parser.parse_args()
    opt = get_opt()
    print(opt, end='\n\n')
    data_path = args.data
    annotation = args.annot
    savefile = args.savefile
    run(data_path, opt, annotation, savefile)
