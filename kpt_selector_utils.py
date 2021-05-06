import os
import sys
import numpy as np
import cv2
import json
import torch
from torchvision import transforms
import importlib
import torch.nn.functional as F

sys.path.insert(1, os.path.join(os.getcwd(), "dvrk_untangling_utils"))

def load_net(path_to_refs, network_dir, use_cuda=1):
    model_module = importlib.import_module(network_dir)
    with open('%s/model-refs.json'%(path_to_refs), 'r') as f:
        ref_annots = json.load(f)
    try:
        num_keypoints = model_module.NUM_KEYPOINTS
    except:
        num_keypoints = None
    net = model_module.Model(num_keypoints=num_keypoints, img_height=model_module.IMG_HEIGHT, img_width=model_module.IMG_WIDTH)

    model_path = os.path.join(network_dir, "checkpoints", ref_annots[network_dir])
    if use_cuda:
        torch.cuda.set_device(0)
        net.load_state_dict(torch.load(os.path.join(model_path, os.listdir(model_path)[0])))
        net = net.cuda()
    else:
        net.load_state_dict(torch.load(os.path.join(model_path, os.listdir(model_path)[0]), map_location='cpu'))
    return net

def preds(network_dir, model, img, use_cuda=1):
    model_module = importlib.import_module(network_dir)
    return model_module.predict(model, img, use_cuda=use_cuda)

def crop_and_resize(box, img, aspect=(640,480)):
    x1, y1, x2, y2 = box
    x_min, x_max = min(x1,x2), max(x1,x2)
    y_min, y_max = min(y1,y2), max(y1,y2)
    box_width = x_max - x_min
    box_height = y_max - y_min

    # resize this crop to be 320x240
    new_width = int((box_height*aspect[0])/aspect[1])
    offset = new_width - box_width
    x_min -= int(offset/2)
    x_max += offset - int(offset/2)

    crop = img[y_min:y_max, x_min:x_max]
    resized = cv2.resize(crop, aspect)
    rescale_factor = new_width/aspect[0]
    offset = (x_min, y_min)
    return resized, rescale_factor, offset

def pixel_crop_to_full(pixels, crop_rescale_factor, x_offset, y_offset):
    global_pixels = pixels * crop_rescale_factor
    global_pixels[:, 0] += x_offset
    global_pixels[:, 1] += y_offset
    return global_pixels

def mask_pixels(image):
    workspace_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, workspace_mask = cv2.threshold(workspace_gray,127,255,cv2.THRESH_BINARY)
    mask_idxs = []
    for i in range(workspace_mask.shape[0]):
        for j in range(workspace_mask.shape[1]):
            if workspace_mask[i,j] > 0:
                mask_idxs.append([j,i])
    mask_idxs = np.array(mask_idxs)
    return mask_idxs, workspace_mask

def gauss_2d(width, height, sigma, u, v, normalize_dist=False):
    x, y = np.meshgrid(np.linspace(0., width, num=width), np.linspace(0., height, num=height))
    x, y = np.transpose(x, (0,1)), np.transpose(y, (0,1))
    gauss = np.exp(-(((x-u)**2 + (y - v)**2)/ ( 2.0 * sigma**2 ) ) )
    # vis_gauss(gauss)
    return gauss.astype(float)

def gauss_2d_batch(width, height, sigma, U, V, normalize_dist=False, single=False):
    if not single:
        U.unsqueeze_(1).unsqueeze_(2)
        V.unsqueeze_(1).unsqueeze_(2)
    X,Y = torch.meshgrid([torch.arange(0., width), torch.arange(0., height)])
    X,Y = torch.transpose(X, 0, 1).cuda(), torch.transpose(Y, 0, 1).cuda()
    G=torch.exp(-((X-U.astype(float))**2+(Y-V.astype(float))**2)/(2.0*sigma**2))
    if normalize_dist:
        return normalize(G).double()
    return G.double()

def normalize(x):
    return F.normalize(x, p=1)

def vis_gauss(gaussians):
    output = cv2.normalize(gaussians, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite('test.png', output)
