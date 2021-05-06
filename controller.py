import os
import sys
import numpy as np
import cv2
import json

sys.path.append(os.getcwd())

from UntanglingMotion import UntanglingMotion
from keypoint_selector import KeypointSelector, crop_and_resize

class dvrkUntangle(object):
    def __init__(self, base_dir, predict=True): 
        self.motion = UntanglingMotion()
        self.kpt_selector = KeypointSelector(base_dir, predict=predict)

        self.depth_thresh = 0.01
        self.bad_action_ctr = 0

        self.throw_outs = 0
        self.idx = 0
        self.pull_offset = 0.02

    def grasp_from_kpt(self, pixels, point, which_arm='PSM2', reid=False, pull_height=0.02):
        # _, grasp_pixel, pull_pixel = [np.array(p) for p in pixels]

        grasp_pixel, pull_pixel = [np.array(p) for p in pixels]
        print("in grasp_from_kpt", pixels)

        print("")
        grasp_position = self.motion.transform_cam2robot(grasp_pixel, point, which_arm=which_arm)
        pull_position = self.motion.transform_cam2robot(pull_pixel, point, which_arm=which_arm)

        grasp_position[2] -= self.depth_thresh
        pull_position[2] = grasp_position[2] + pull_height # drop point at slightly higher depth than pull point

        print("grasp position:", grasp_position)
        print("pull position:", pull_position)
        return grasp_position, pull_position, grasp_pixel

    def hold_from_kpt(self, hold_pixel, point, which_arm="PSM1"):
        print("HOLD PIXEL", hold_pixel)
        hold_position = self.motion.transform_cam2robot(hold_pixel, point, which_arm=which_arm)
        hold_position[2] -= self.depth_thresh
        print("hold position:", hold_position)
        return hold_position, hold_pixel

    def untangle_to_throwout(self, img, throw_pixel=[1780,550]):
        pixels, grasp_rotation_r, point, self.zc = self.kpt_selector.reid(image=img, right=True, idx=self.idx, plot=True)
        
        if pixels is None:
            return True # no more cables in scene

        grasp_pixel = pixels[0]
        pixels = [grasp_pixel, throw_pixel]
        self.idx += 1
        grasp_position_r, throw_position, grasp_pixel_r = self.grasp_from_kpt(pixels, point, reid=True, which_arm='PSM1', pull_height=0.04)
        pull_rotation_r = grasp_rotation_r # rotations should be the same for PSM2 arm points

        self.motion.move_origin()
        if not self.user_check(): return
        self.motion.grasp_psm1(grasp_position_r, grasp_rotation_r, reid=True)

        pixels, grasp_rotation_l, point, self.zc = self.kpt_selector.reid(right=False, idx=self.idx, plot=True)
        self.idx += 1
        grasp_position_l, pull_position_l, grasp_pixel_l = self.grasp_from_kpt(pixels, point, reid=True)
        pull_rotation_l = grasp_rotation_l 

        if not self.user_check(): return
        self.motion.grasp_psm2(grasp_position_l, grasp_rotation_l, reid=True, thresh=False, slight_grasp=True)

        self.motion.lift_psm1(grasp_position_r, grasp_rotation_r) # @Priya
        self.motion.move_origin(open_psm1=False, open_psm2=True, move_psm2=False, move_psm1=True)
        self.motion.pull_psm1(throw_position, pull_rotation_r, reid=True)
        self.motion.release()
        self.motion.move_origin()

        return

    def reid_both(self):
        pixels, grasp_rotation_r, point, self.zc = self.kpt_selector.reid(right=True, idx=self.idx, plot=True)
        self.idx += 1
        grasp_position_r, pull_position_r, grasp_pixel_r = self.grasp_from_kpt(pixels, point, reid=True, which_arm='PSM1')
        # _, grasp_rotation_r = rotations
        pull_rotation_r = grasp_rotation_r # rotations should be the same for PSM2 arm points

        self.motion.move_origin()
        if not self.user_check(): return self.to_origin()
        self.motion.grasp_psm1(grasp_position_r, grasp_rotation_r, reid=True)

        pixels, grasp_rotation_l, point, self.zc = self.kpt_selector.reid(right=False, idx=self.idx, plot=True)
        self.idx += 1
        grasp_position_l, pull_position_l, grasp_pixel_l = self.grasp_from_kpt(pixels, point, reid=True)
        # _, grasp_rotation_l = rotations
        pull_rotation_l = grasp_rotation_l # rotations should be the same for PSM2 arm points

        if grasp_pixel_r[0] - grasp_pixel_l[0] < 0:
            self.to_origin()
            return

        if not self.user_check(): return self.to_origin()
        self.motion.grasp_psm2(grasp_position_l, grasp_rotation_l, reid=True)
        self.motion.lift_psm2(grasp_position_l, grasp_rotation_l) # @Priya
        self.motion.lift_psm1(grasp_position_r, grasp_rotation_r) # @Priya

        self.motion.pull_psm2(pull_position_l, pull_rotation_l, reid=True)
        self.motion.pull_psm1(pull_position_r, pull_rotation_r, reid=True)

        self.motion.lowered_release(pull_position_r, pull_position_l, rot1=pull_rotation_r, rot2=pull_rotation_l)
        self.motion.move_origin()

    def reidemeister(self):
        self.motion.move_origin()
        self.reid_both()

    def user_check(self):
        execute = input("Execute action?")
        if execute in ["n", "q"]:
            self.motion.move_origin()
            self.bad_action_ctr += 1
            return False
        return True

    def undo(self):
        pixels, rotations, point, self.zc = self.kpt_selector.undo(idx=self.idx)

        untangle_from_right = pixels[0][0] > pixels[1][0] # is the hold to the right of the pull
        print("UNTANGLE FROM RIGHT", untangle_from_right)
        which_arm_grasp = "PSM2" if untangle_from_right else "PSM1"
        which_arm_hold = "PSM1" if untangle_from_right else "PSM2"

        pull_drop_pixels = pixels[1:]
        self.idx += 1
        grasp_position, pull_position, grasp_pixel = self.grasp_from_kpt(pull_drop_pixels, point, which_arm=which_arm_grasp)

        hold_rotation, grasp_rotation = rotations
        pull_rotation = grasp_rotation # rotations should be the same for PSM2 arm points
        print("grasp rotation:", grasp_rotation)

        self.motion.move_origin()
        if not self.user_check(): return None, None, None
        if untangle_from_right:
            self.motion.grasp_psm2(grasp_position, grasp_rotation)
        else: 
            self.motion.grasp_psm1(grasp_position, grasp_rotation)

        pixels, _, point, _ = self.kpt_selector.undo(idx=self.idx)
        self.idx += 1
        hold_pixel = pixels[0]
        hold_position, hold_pixel = self.hold_from_kpt(hold_pixel, point, which_arm=which_arm_hold)

        print("hold rotation:", hold_rotation)

        dx = grasp_pixel[0] - hold_pixel[0]
        dy = grasp_pixel[1] - hold_pixel[1]
        
        if untangle_from_right:
            if hold_pixel[0] - grasp_pixel[0] < 0 or np.linalg.norm(np.array([dx,dy])) < 7:
                self.bad_action_ctr += 1
                print("Predicted hold too close or left of pull")
                self.motion.release()
                self.motion.move_origin()
                return None, None, None
            else:
                assert(which_arm_grasp == "PSM2")
                assert(which_arm_hold == "PSM1")
                if not self.user_check():
                    return None, None, None
                self.bad_action_ctr = 0
                self.motion.hold_psm1(hold_position, hold_rotation) # PSM1
                self.motion.pull_psm2(pull_position, pull_rotation)
        else:
            if hold_pixel[0] - grasp_pixel[0] > 0 or np.linalg.norm(np.array([dx,dy])) < 7:
                self.bad_action_ctr += 1
                print("Predicted hold too close or right of pull")
                self.motion.release()
                self.motion.move_origin()
                return None, None, None
            else:
                assert(which_arm_grasp == "PSM1")
                assert(which_arm_hold == "PSM2")
                if not self.user_check(): return None, None, None
                self.bad_action_ctr = 0
                self.motion.hold_psm2(hold_position, hold_rotation) # PSM2
                self.motion.pull_psm1(pull_position, pull_rotation)
        self.motion.release()
        self.motion.move_origin()

        action_vec = [dx, dy, 6] # 6 is arbitrary for dz
        action_vec /= np.linalg.norm(action_vec)

        return grasp_pixel, hold_pixel, action_vec

    def recenter(self, reid=False):
        pixels, grasp_rotation, point, self.zc = self.kpt_selector.rope_center_pixel_predict(plot=True)
        use_psm1 = pixels[0][0] > self.kpt_selector.center_drop[0]
        self.idx += 1

        self.motion.move_origin()
        if use_psm1:
            grasp_position, drop_position, grasp_pixel = self.grasp_from_kpt(pixels, point, which_arm='PSM1')
            pull_rotation = grasp_rotation # rotations should be the same for PSM1 arm points
            if not self.user_check(): return None, None, None
            self.motion.grasp_psm1(grasp_position, grasp_rotation, reid=True)
            self.motion.pull_psm1(drop_position, pull_rotation)
        else:
            grasp_position, drop_position, grasp_pixel = self.grasp_from_kpt(pixels, point, which_arm='PSM2')
            pull_rotation = grasp_rotation # rotations should be the same for PSM1 arm points
            if not self.user_check(): return None, None, None
            self.motion.grasp_psm2(grasp_position, grasp_rotation, reid=True)
            self.motion.pull_psm2(drop_position, pull_rotation)

        self.motion.release()
        self.motion.move_origin()

    def undone_check(self):
        return self.kpt_selector.term_predict()

    def to_origin(self):
        self.motion.move_origin()

    def rope_in_workspace(self):
        return self.kpt_selector.check_untangling()

    def check_recenter(self):
        return self.kpt_selector.check_recenter()

    def check_gripper_stuck(self):
        self.motion.move_origin(check=True)
        stuck = self.kpt_selector.check_gripper_stuck(plot=True)
        print("Stuck gripper?:", stuck)

        if stuck=='PSM1':
            pixels = np.array([self.kpt_selector.center_drop, self.kpt_selector.center_drop])
            pull_rotation = 0
            point = self.kpt_selector.full_img_point
            grasp_position, pull_position, _ = self.grasp_from_kpt(pixels, point, which_arm='PSM1')
            pull_position[2] -= self.pull_offset - 0.01
            self.motion.move_origin(open_psm1=False, open_psm2=False)
            if not self.user_check(): return
            self.motion.grasp_psm1(grasp_position, pull_rotation, reid=True, open=False)
            self.motion.pull_psm1(pull_position, pull_rotation, reid=True)
            
            # use PSM2 to pin down other side of rope
            grasp_drop_pixels, rot, self.full_img_point, self.zc = self.kpt_selector.reid(right=False, idx=self.idx)
            inner_point, _ = grasp_drop_pixels

            grasp_position, _, _ = self.grasp_from_kpt(grasp_drop_pixels, point, which_arm='PSM2')
            if not self.user_check(): return
            self.motion.grasp_psm2(grasp_position, rot)
            #self.motion.move_origin_psm1()
            self.motion.move_origin(open_psm2=False, move_psm2=False)
            
            self.motion.release()
            self.motion.move_origin()
        elif stuck=='PSM2':
            pixels = np.array([self.kpt_selector.center_drop, self.kpt_selector.center_drop])
            pull_rotation = 0
            point = self.kpt_selector.full_img_point
            grasp_position, pull_position, _ = self.grasp_from_kpt(pixels, point, which_arm='PSM2')
            pull_position[2] -= self.pull_offset - 0.01
            self.motion.move_origin(open_psm1=False, open_psm2=False)
            if not self.user_check(): return
            self.motion.grasp_psm2(grasp_position, pull_rotation, reid=True, open=False)
            self.motion.pull_psm2(pull_position, pull_rotation, reid=True)
            
            # use PSM1 to pin down other side of rope
            grasp_drop_pixels, rot, self.full_img_point, self.zc = self.kpt_selector.reid(right=True, idx=self.idx)
            inner_point, _ = grasp_drop_pixels

            grasp_position, _, _ = self.grasp_from_kpt(grasp_drop_pixels, point, which_arm='PSM1')
            if not self.user_check(): return
            self.motion.grasp_psm1(grasp_position, rot)
            #self.motion.move_origin_psm2()
            self.motion.move_origin(open_psm1=False, move_psm1=False)
            
            self.motion.release()
            self.motion.move_origin()
        return stuck

def run_untangling_rollout(policy, max_actions=20):

    if not os.path.exists('rollout'):
        os.mkdir('rollout')
    throwout_actions = 0
    undo_actions = 0
    recenter_actions = 0
    recovery_counter = 0
    limit = 30
    policy.to_origin()
    policy.reidemeister()

    ropes_left = policy.rope_in_workspace()
    print("ROPES LEFT?", ropes_left)

    while ropes_left:
        print("Action counters: consec. bad actions: %d, undo actions: %d, recenter: %d, recovery: %d, throwout: %d"%(policy.bad_action_ctr, undo_actions, recenter_actions, recovery_counter, throwout_actions))
        undone, img_throwout = policy.undone_check()
        recenter = policy.check_recenter()
        print("RECENTER?", recenter)
        if recenter:
            policy.recenter(reid=True)
            recenter_actions += 1
        elif img_throwout is not None or (undo_actions >= limit) or policy.bad_action_ctr >= 3:
            print("throwing out")
            undone_check = policy.untangle_to_throwout(img_throwout) 
            throwout_actions += 1
            if undone_check is not None:
                return
        else:
            pull, hold, action_vec = policy.undo()
            undo_actions += 1

        ropes_left = policy.rope_in_workspace()
        if ropes_left:
            stuck = policy.check_gripper_stuck()
            while stuck is not None:
                recovery_counter += 1
                stuck = policy.check_gripper_stuck()

    policy.to_origin()
    print("FINAL ACTION COUNTERS: consec. bad actions: %d, undo actions: %d, recenter: %d, recovery: %d, throwout: %d"%(policy.bad_action_ctr, undo_actions, recenter_actions, recovery_counter, throwout_actions))

if __name__ == "__main__":
    BASE_DIR = os.getcwd()
    policy = dvrkUntangle(BASE_DIR)
    policy.to_origin()

    run_untangling_rollout(policy)
    #stuck = policy.check_gripper_stuck()
