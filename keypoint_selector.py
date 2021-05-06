import cv2
import os
import sys
import numpy as np
import torch
from ZividCapture import ZividCapture
from pixel_selector import PixelSelector
BASE_DIR = os.getcwd()
from torchvision import transforms, utils
sys.path.append(BASE_DIR)
from kpt_selector_utils import load_net, preds, gauss_2d, gauss_2d_batch, pixel_crop_to_full, crop_and_resize

sys.path.insert(0, os.path.join(BASE_DIR, "dvrk_untangling_utils"))
from adaptive_crop import contour_crop
from project_preds import project_pred
from brighten_contrast import brighten_contrast
from image_empty import check_empty
from pca_grasp import run_pca, get_grasping_crop
from sklearn.neighbors import NearestNeighbors

os.environ["CUDA_VISIBLE_DEVICES"]="0"

class KeypointSelector(object):
    def __init__(self, base_dir, predict=True):
        self.zc = ZividCapture(which_camera="overhead") 
        self.zc_on = False
        self.predict_ph = predict
        self.pca_crop_size = (25, 25)
        self.refs_dir = BASE_DIR
        self.base_dir = base_dir
        self.transform = transforms.Compose([transforms.ToTensor()]) 

        self.models = {}
        self.model_dirs = ["reidemeister", "kpt_conditioned", "grasp_rot_off_net", "termination", "kpt_hulk"]

        self.crop_width = 400
        self.crop_height = 300
        self.network_inp_size = (640,480)

        # Cropping dims for full workspace, used in check gripper stuck
        self.wkspace_x = 550
        self.wkspace_y = 250
        self.wkspace_w =  950
        self.wkspace_h = 420
        self.left_grip_home_px = [580,550] 
        self.right_grip_home_px = [1475,545]

        # Cropping dims for untangling region
        self.untangle_x = 825
        self.untangle_y = 275
        self.untangle_crop_w = 470
        self.untangle_crop_h = 480

        self.reid_right = False
        self.reid_left = False
        self.recenter = False

        # Reidemeister drop locations
        self.right_drop = [1200, 475] 
        self.left_drop = [900, 525] 
        self.center_drop = [1000, 510]

        self.load_models()

    def img_cap(self, stage=False, plot=False, wide_crop=False):
        self.zivid_cap(plot=plot)
        return self.img_crop(wide_crop=wide_crop)

    def get_zc(self):
        if not self.zc_on:
            self.zc.start()
        self.zc_on = True
        return self.zc

    def zivid_cap(self, plot=False):
        if not self.zc_on:
            self.zc.start()
        self.zc_on = True

        self.img_color, self.img_depth, self.img_point = self.zc.capture_3Dimage(color='BGR')
        if plot:
            cv2.imshow("", self.img_color)
            cv2.waitKey(0)
        return self.img_color

    def find_box(self, image, stage=False, wide_crop=False):
        crop_size = (self.crop_width, self.crop_height)
        if wide_crop:
            box, _ = contour_crop(image, crop_size, wkspace_x=self.wkspace_x, \
                                                    wkspace_y=self.wkspace_y, \
                                                    wkspace_w=self.wkspace_w, \
                                                    wkspace_h=self.wkspace_h, plot=False)
        else:
            box, _ = contour_crop(image, crop_size, wkspace_x=self.untangle_x, \
                                                    wkspace_y=self.untangle_y, \
                                                    wkspace_w=self.untangle_crop_w, \
                                                    wkspace_h=self.untangle_crop_h, plot=False)
        return box

    def img_crop(self, stage=False, image=None, point=True, depth=True, wide_crop=False):
        if image is None:
            image = self.img_color

        self.full_img_color = image
        self.full_img_point = self.img_point
        box = self.find_box(image, stage=stage, wide_crop=wide_crop)
        self.img_color, self.rescale_factor, self.offset = crop_and_resize(box, image, aspect=self.network_inp_size)
        try:
            self.img_depth, _, _ = crop_and_resize(box, self.img_depth, aspect=self.network_inp_size)
            self.img_point, _, _ = crop_and_resize(box, self.img_point, aspect=self.network_inp_size)
        except:
            pass
        self.img_color = brighten_contrast(self.img_color)
        return self.img_color

    def check_untangling(self):
        self.img_cap()
        return check_empty(self.img_color)

    def get_drop(self, hold, pull):
        dx = pull[0] - hold[0]
        dy = pull[1] - hold[1]
        if not dy:
            dy = 4
        if not dx:
            dx = 4
        #drop_length = 50
        drop_length = 85
        drop_dx = drop_length * dx / np.linalg.norm([dx, dy])
        drop_dy = drop_length * dy / np.linalg.norm([dx, dy])
        return np.array([pull[0]+drop_dx, pull[1]+drop_dy])

    def pixel_select(self):
        self.img_cap()
        pixel_selector = PixelSelector()
        # pixels should be selected in the order of hold, pull, drop
        self.preds = pixel_selector.run(self.img_color.copy())
        if len(self.preds) == 2:
            self.preds.append(self.get_drop(self.preds[0], self.preds[1]))

        img = self.img_color.copy()
        self.preds[0] = project_pred(img, self.preds[0], plot=True)
        self.preds[1] = project_pred(img, self.preds[1], plot=True)
        return self.preds

    # JENN TESTED
    def condition_heatmap(self, img, pixel, gauss_sigma=8):
        img_height, img_width, _ = img.shape
        img = self.transform(img)
        given_gauss = gauss_2d_batch(img_width, img_height, gauss_sigma, pixel[0], pixel[1], single=True)
        given_gauss = torch.unsqueeze(given_gauss, 0).cuda()
        combined = torch.cat((img.cuda().double(), given_gauss), dim=0).float()
        return combined

    def load_models(self):
        for m in self.model_dirs:
            self.models[m] = load_net(self.base_dir, m)
        return self.models

    # JENN: FIXED
    def img_predict(self, img, model_index=0):
        model_dir = self.model_dirs[model_index]
        self.preds, self.pred_class = preds(model_dir, self.models[model_dir], img)
        if (model_index == 1):
            img = img.detach().cpu().numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, :3]*255          
        if (model_index == 0 or model_index == 1 or model_index==4):    
            proj_pred0 = project_pred(img, self.preds[0])
            proj_pred1 = project_pred(img, self.preds[1])
            if proj_pred0 is not None:
                self.preds[0] = proj_pred0
            if proj_pred1 is not None:
                self.preds[1] = proj_pred1
            if proj_pred0 is None or proj_pred1 is None:
                self.preds = [None, None]
        return self.preds

    def rescale_pixels(self, pixels):
        new_2D = []
        for p in pixels:
            if p is not None:
                rescaled_p = pixel_crop_to_full(np.array([p]), self.rescale_factor, self.offset[0], self.offset[1])[0] 
                new_2D.append([int(r) for r in rescaled_p])
            else:
                new_2D.append(None)
        return np.array(new_2D)

    def get_3Dpixels(self):
        self.pixels3D = []
        depths = []
        for p in self.pixels2D_crop:
            depth = self.img_point[p[1],p[0], 2] # FIX THIS
            depths.append(depth)
        for i in range(len(self.pixels2D)):
            p = self.pixels2D[i]
            self.pixels3D.append([p[0], p[1], depths[i]])

    def check_rotation(self, rot):
        if rot < -90:
            rot += 180
        if rot > 90:
            rot -= 180
        return rot

    def get_rot_off_grasping_net(self, grasp_pixel):
        grasp_crop_size = (40,40)
        upscaled_dims = (200,200)
        crop = get_grasping_crop(self.full_img_color.copy(), tuple(grasp_pixel), grasp_crop_size, plot=False, inner_vert=False)
        rot_learned, (off_x,off_y) = self.img_predict(crop, model_index=2) # grasping net rot off
        off_x = int(grasp_crop_size[0]/upscaled_dims[0]*off_x)
        off_y = int(grasp_crop_size[1]/upscaled_dims[1]*off_y)

        global_x, global_y = grasp_pixel
        new_grasp_pixel = (global_x + off_x, global_y + off_y)

        rot_learned = self.check_rotation(np.degrees(rot_learned))
        return rot_learned, new_grasp_pixel

    def plot_pixels(self, image, pixels, save_dir=None, filename=None, plot=False):
        if pixels[0] is None:
            return
        plot_img = image.copy()
        colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0)]
        colors = colors[:len(pixels)]
        for p, c in zip(pixels, colors):
            plot_img = cv2.circle(plot_img, tuple(p), 5, c, 2)

        if save_dir is not None:
            filename = "test.png" if filename is None else filename
            cv2.imwrite(os.path.join(BASE_DIR, save_dir, filename), plot_img)

        if plot:
            cv2.imshow("", plot_img)
            cv2.waitKey(0)

    def term_predict(self):
        img = self.img_cap()
        self.load_models()
        done = self.img_predict(img, model_index=3)[0]
        return done, img if done else None

    def get_rope_left(self, idx=0, plot=False):
        img = self.img_cap()
        crop_size = (self.crop_width, self.crop_height)
        _, mask_and_center = contour_crop(self.full_img_color, crop_size, plot=False)
        mask_pixels, _, wkspace_x, wkspace_y = mask_and_center

        rope_pixel_crop = sorted(mask_pixels, key=lambda p: p[0])[0]
        rope_pixel = pixel_crop_to_full(np.array([rope_pixel_crop]), 1, wkspace_x, wkspace_y)[0]

        rot, inner_point = self.get_rot_off_grasping_net(rope_pixel) # GRASPING NET WITH OFFSET
        self.plot_pixels(self.full_img_color, [inner_point], plot=plot)

        grasp_drop_pixels = [inner_point, self.center_drop]
        return grasp_drop_pixels, rot, self.full_img_point, self.zc

    def get_rope_right(self, idx=0, plot=False, recenter=False):
        img = self.img_cap()
        crop_size = (self.crop_width, self.crop_height)
        _, mask_and_center = contour_crop(self.full_img_color, crop_size, plot=False)
        mask_pixels, _, wkspace_x, wkspace_y = mask_and_center

        rope_pixel_crop = sorted(mask_pixels, key=lambda p: p[0])[-1]
        rope_pixel = pixel_crop_to_full(np.array([rope_pixel_crop]), 1, wkspace_x, wkspace_y)[0]

        rot, inner_point = self.get_rot_off_grasping_net(rope_pixel) # GRASPING NET WITH OFFSET
        self.plot_pixels(self.full_img_color, [inner_point], plot=plot)

        grasp_drop_pixels = [inner_point, self.center_drop] if not recenter else [inner_point, self.right_recenter_drop]
        return grasp_drop_pixels, rot, self.full_img_point, self.zc

    def rope_center_pixel_predict(self, plot=True):
        img = self.img_cap()
        crop_gray = cv2.cvtColor(self.img_color, cv2.COLOR_BGR2GRAY)
        _, crop_mask = cv2.threshold(crop_gray,60,255,cv2.THRESH_BINARY)
        mask_idxs = np.nonzero(crop_mask)
        ys, xs = mask_idxs
        mu_y = int(np.mean(ys))
        mu_x = int(np.mean(xs))
        rope_center = (mu_x, mu_y)

        neigh = NearestNeighbors(1, 0)
        pixels = np.vstack((xs, ys)).T
        neigh.fit(pixels)
        match_idxs = neigh.kneighbors([rope_center], 1, return_distance=False).squeeze()
        projected_rope_center = tuple(pixels[match_idxs])

        self.preds = [projected_rope_center, self.center_drop]
        self.preds = [[int(p[0]), int(p[1])] for p in self.preds if p is not None]
        if plot:
            self.plot_pixels(self.img_color, self.preds, plot=True)

        rescaled = self.preds
        rescaled[0] = self.rescale_pixels([self.preds[0]])[0]

        rot, inner_point = self.get_rot_off_grasping_net(rescaled[0]) # GRASPING NET WITH OFFSET
        rescaled[0] = inner_point
        return rescaled, rot, self.full_img_point, self.zc


    def check_recenter(self, plot=False):
        img = self.img_cap()
        crop_size = (self.crop_width, self.crop_height)
        _, mask_and_center = contour_crop(self.full_img_color, crop_size, wkspace_x=750, wkspace_y=300)
        mask_pixels, crop_center, wkspace_x, wkspace_y = mask_and_center

        nbrs = NearestNeighbors(n_neighbors=1).fit(mask_pixels)
        _, rope_center_idx = nbrs.kneighbors(np.array([crop_center]))
        rope_center_crop = mask_pixels[rope_center_idx[0][0]]
        rope_center = pixel_crop_to_full(np.array([rope_center_crop]), 1, wkspace_x, wkspace_y)[0]
        center_x, center_y = rope_center

        dist_center = np.linalg.norm(np.array([center_x, center_y]) - np.array(self.center_drop))
        print("Center dist", dist_center)
        thresh_center_offset = 155

        left_extremity = [790, 550]
        right_extremity = [1150, 550]

        if center_x < left_extremity[0] or center_x > right_extremity[0] or dist_center > thresh_center_offset:
            recenter = True
        else:
            recenter = False
        return recenter

    def check_gripper_stuck(self, plot=True):
        img = self.img_cap(wide_crop=True)
        crop_size = (self.crop_width, self.crop_height)

        box, (dist_left, dist_right) = contour_crop(self.full_img_color, crop_size, wkspace_x=self.wkspace_x, \
                                                wkspace_y=self.wkspace_y, \
                                                wkspace_w=self.wkspace_w, \
                                                wkspace_h=self.wkspace_h, \
                                                psm2_px=self.left_grip_home_px, \
                                                psm1_px=self.right_grip_home_px, \
                                                plot=False)

        box_center_x = 0.5*(box[0]+box[2])
        box_center_y = 0.5*(box[1]+box[3])

        dist_center = np.linalg.norm(np.array([box_center_x, box_center_y]) - np.array(self.center_drop))
        print("Dists: (left, right, center):", dist_left, dist_right, dist_center)
        print("Box:", box_center_x, box_center_y)
        thresh_gripper_offset = 30

        if dist_left < thresh_gripper_offset:
            stuck = 'PSM2'
        elif dist_right < thresh_gripper_offset:
            stuck = 'PSM1' 
        else:
            stuck = None
        return stuck

    def nn_from_pixel_mask(self, img_mask, px):
        neigh = NearestNeighbors(1, 0)
        ys, xs = np.where(img_mask > 0)
        pixels = np.vstack((xs, ys)).T
        neigh.fit(pixels)
        match_idxs = neigh.kneighbors([px], 1, return_distance=False).squeeze()
        nn = pixels[match_idxs]
        return nn

    def reid(self, image=None, idx=0, stage=False, right=False, plot=True):
        img = image
        if image is None:
            img = self.img_cap()
        cv2.imshow('img', img)
        cv2.waitKey(0)

        pred_idx = 0 if right else 1

        endpoint_pred = np.array([self.img_predict(img, model_index=0)[pred_idx]]) # reidemeister
        
        if endpoint_pred[0] is None:
            return None, None, None, None

        rescaled_pred = self.rescale_pixels(endpoint_pred)
        self.plot_pixels(self.img_color, endpoint_pred, plot=plot)
        self.plot_pixels(self.full_img_color, rescaled_pred, plot=plot)

        rot, inner_point = self.get_rot_off_grasping_net(rescaled_pred[0]) # GRASPING NET WITH OFFSET
        points = [inner_point, self.right_drop] if right else [inner_point, self.left_drop] # [grasp, drop]

        return points, rot, self.full_img_point, self.zc

    def undo(self, idx=0, plot=True):

        img = self.img_cap()

        pin_pull_pred = self.img_predict(img, model_index=4) # kpt_hulk
        pin_pull_pred = pin_pull_pred[:2]
        pin_pred, pull_pred = pin_pull_pred

        drop = [int(p) for p in self.get_drop(pin_pred, pull_pred)]
        crop_points = [pin_pred, pull_pred, drop]

        rescaled_pred = self.rescale_pixels(crop_points)
        self.plot_pixels(self.img_color, crop_points, plot=plot, save_dir="rollout", filename="preds%d.png"%idx)
        self.plot_pixels(self.full_img_color, rescaled_pred, plot=plot)

        hold_rot, rescaled_pred[0] = self.get_rot_off_grasping_net(rescaled_pred[0]) # GRASPING NET WITH OFFSET
        grasp_rot, rescaled_pred[1] = self.get_rot_off_grasping_net(rescaled_pred[1])
        rots = [hold_rot, grasp_rot]

        return rescaled_pred, rots, self.full_img_point, self.zc
    
if __name__ == '__main__':
    BASE_DIR = os.getcwd()
    kp_selector = KeypointSelector(BASE_DIR, zivid=True, crop=True)
    kp_selector.rope_center_pixel_predict( plot=True)
