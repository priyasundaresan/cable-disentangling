from motion.dvrkDualArm import dvrkDualArm
import os
import sys
sys.path.append(os.getcwd())
from keypoint_selector import KeypointSelector
import utils.CmnUtil as U
import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors

def closest_3D_pixel(img_point, point_cloud):
    y_dim, x_dim, _ = point_cloud.shape
    point_cloud_pt = np.array(point_cloud[int(img_point[1]), int(img_point[0])]) * 0.001

    if not np.isnan(point_cloud_pt[0]) and not np.isnan(point_cloud_pt[1]):
    	return point_cloud_pt

    non_nan_ys, non_nan_xs  = np.where(np.logical_not(np.isnan(point_cloud[:,:,2])))
    global_non_nan_pixels = np.vstack((non_nan_ys, non_nan_xs)).T
    nbrs = NearestNeighbors(n_neighbors=1).fit(global_non_nan_pixels)
    _, closest_non_nan_idx = nbrs.kneighbors(np.array([img_point[1], img_point[0]]).reshape(1, -1))
    closest_non_nan_px = global_non_nan_pixels[closest_non_nan_idx].squeeze()
    point_cloud_pt = np.array(point_cloud[closest_non_nan_px[0], closest_non_nan_px[1]])* 0.001
    return point_cloud_pt

class UntanglingMotion:
	def __init__(self):
		# instances
		self.dvrk = dvrkDualArm()
		self.top_down = False

		# load transform
		self.Trc1 = np.load('../calibration_files/Trc_overhead_PSM1.npy')
		self.Trc2 = np.load('../calibration_files/Trc_overhead_PSM2.npy')

		# Motion variables
		self.lift_height = -0.0775
		self.grasp_above_offset = 0.02

		self.psm1_jaw_angle_offset = np.deg2rad(20) # red needle driver
		self.psm2_jaw_angle_offset = np.deg2rad(5)

		self.jaw_opening = [np.deg2rad(50)]
		self.jaw_closing = [np.deg2rad(3)]

		self.psm1_jaw_opening = [0.5]
		self.psm1_jaw_closing = [-0.1]
		self.hold_jaw_opening = self.psm1_jaw_opening
		self.hold_jaw_closing = [0.4]
		self.psm2_jaw_opening = [0.7]
		self.psm2_jaw_closing = [-0.05]

		approach_angle = 30
		self.psm1_lat_rot = np.deg2rad(20-approach_angle) # red needle driver
		self.psm2_lat_rot = np.deg2rad(approach_angle)
		self.axis_rot = np.deg2rad(20)
		
		reid_approach_angle = 0
		self.psm1_lat_rot_reid = np.deg2rad(-reid_approach_angle) # red needle driver
		self.psm2_lat_rot_reid = np.deg2rad(reid_approach_angle)

		self.psm2_grasp_rot_offset = 20

		# define origin points
		self.origin_depth = -0.06
		self.pos_org1 = [0.040, 0.0, self.origin_depth]
		self.rot_org1 = self.inclined_orientation(np.deg2rad(0), self.psm1_lat_rot)
		self.pos_org2 = [-0.050, 0.0, self.origin_depth]
		self.rot_org2 = self.inclined_orientation(np.deg2rad(0+self.psm2_grasp_rot_offset), self.psm2_lat_rot)

		self.pos_check_org1 = [0.010, 0.0, self.origin_depth]
		self.rot_org1 = self.inclined_orientation(np.deg2rad(0), self.psm1_lat_rot) # !!! USED TO BE THIS
		self.pos_check_org2 = [-0.010, 0.0, self.origin_depth]
		self.rot_org2 = self.inclined_orientation(np.deg2rad(0), self.psm2_lat_rot)

		self.pos_org_staging2 = [-0.05, 0.05, self.origin_depth]

		self.psm1_min_depth = -0.1390
		self.psm2_min_depth = -0.1390

	def transform_cam2robot(self, img_point, point_cloud, which_arm='PSM1'):  # input: pixel point
		p_cam = closest_3D_pixel(img_point, point_cloud)
		if which_arm == 'PSM1':
			Trc = self.Trc1
		else:
			Trc = self.Trc2
		R = Trc[:3, :3]
		t = Trc[:3, -1]
		return R.dot(p_cam.T).T + t.T

	def move_origin_check(self, open=True, close=True):
		if open: self.dvrk.set_jaw(jaw1=self.psm1_jaw_opening, jaw2=self.psm2_jaw_opening)
		self.dvrk.set_pose(pos1=self.pos_check_org1, rot1=self.rot_org1, pos2=self.pos_check_org2, rot2=self.rot_org2)
		if close: self.dvrk.set_jaw(jaw1=self.psm1_jaw_closing, jaw2=self.psm2_jaw_closing)

	def move_origin_check_psm1(self, open=True, close=True):
		if open: self.dvrk.set_jaw(jaw1=self.psm1_jaw_opening)
		self.dvrk.set_pose(pos1=self.pos_check_org1, rot1=self.rot_org1)
		if close: self.dvrk.set_jaw(jaw1=self.psm1_jaw_closing)

	def move_origin_check_psm2(self, open=True, close=True):
		if open: self.dvrk.set_jaw(jaw2=self.psm2_jaw_opening)
		self.dvrk.set_pose(pos2=self.pos_check_org2, rot2=self.rot_org2)
		if close: self.dvrk.set_jaw(jaw2=self.psm2_jaw_closing)

	def move_origin(self, open_psm1=True, open_psm2=True, move_psm1=True, move_psm2=True, check=False):
		if check:
			self.move_origin_check(open=False, close=True)
		else:
			if open_psm1:
				self.dvrk.set_jaw(jaw1=self.psm1_jaw_opening)
			if open_psm2:
				self.dvrk.set_jaw(jaw2=self.psm2_jaw_opening)
			if move_psm1:
				self.dvrk.set_pose(pos1=self.pos_org1, rot1=self.rot_org1)
			if move_psm2:
				self.dvrk.set_pose(pos2=self.pos_org2, rot2=self.rot_org2)
			self.dvrk.set_jaw(jaw1=self.psm1_jaw_closing, jaw2=self.psm2_jaw_closing)


	def get_rot_offset(self, rot_grasp, approach_angle, longitude=0):
		gripper_length = 0.006 # in meters
		z = gripper_length*np.sin(approach_angle)
		x_offset = z*np.cos(rot_grasp)
		y_offset = z*np.sin(rot_grasp)
		return x_offset, y_offset

	def hold_psm1(self, pos1, rot1):	# using PSM1
		#pos = [pos1[0], pos1[1], self.height_ready]
		pos = [pos1[0], pos1[1], pos1[2]+self.grasp_above_offset]
		rot = self.inclined_orientation(np.deg2rad(rot1), self.psm1_lat_rot)

		x_offset, y_offset = self.get_rot_offset(rot1, self.psm1_lat_rot)
		pos[0] += x_offset
		pos[1] -= y_offset

		self.dvrk.set_pose(pos1=pos, rot1=rot)
		self.dvrk.set_jaw(jaw1=self.hold_jaw_opening)

		pos = [pos1[0], pos1[1], max(pos1[2], self.psm1_min_depth)]
		rot = self.inclined_orientation(np.deg2rad(rot1), self.psm1_lat_rot)
		self.dvrk.set_pose(pos1=pos, rot1=rot)
		self.dvrk.set_jaw(jaw1=self.hold_jaw_closing)

	def hold_psm2(self, pos2, rot2):
        #pos = [pos2[0], pos2[1], self.height_ready]
		pos = [pos2[0], pos2[1], pos2[2]+self.grasp_above_offset]
		rot = self.inclined_orientation(np.deg2rad(rot2 + self.psm2_grasp_rot_offset), self.psm2_lat_rot)
		self.dvrk.set_pose(pos2=pos, rot2=rot)
		self.dvrk.set_jaw(jaw2=self.psm2_jaw_opening)

		# go down toward the knot & close jaw
		pos = [pos2[0], pos2[1], max(pos2[2], self.psm2_min_depth)]
		rot = self.inclined_orientation(np.deg2rad(rot2+ self.psm2_grasp_rot_offset), self.psm2_lat_rot)
		self.dvrk.set_pose(pos2=pos, rot2=rot)
		self.dvrk.set_jaw(jaw2=self.hold_jaw_closing)


	def pull_psm1(self, pos1, rot1, reid=False):	# using PSM1
		# pull knot
		pos = [pos1[0], pos1[1], max(pos1[2], self.psm1_min_depth)]
		approach_angle = self.psm1_lat_rot_reid if reid else self.psm1_lat_rot
		rot = self.inclined_orientation(np.deg2rad(rot1), approach_angle)
		self.dvrk.set_pose(pos1=pos, rot1=rot)

	def grasp_psm1(self, pos1, rot1, reid=False, thresh=True, open=True):	# using PSM1
		#pos = [pos1[0], pos1[1], self.height_ready]
		pos = [pos1[0], pos1[1], pos1[2]+self.grasp_above_offset]
		approach_angle = self.psm1_lat_rot_reid if reid else self.psm1_lat_rot
		rot = self.inclined_orientation(np.deg2rad(rot1), approach_angle)

		x_offset, y_offset = self.get_rot_offset(rot1, approach_angle)
		pos[0] += x_offset
		pos[1] -= y_offset

		if open:
			self.dvrk.set_jaw(jaw1=self.psm1_jaw_opening)
		self.dvrk.set_pose(pos1=pos, rot1=rot)
		self.dvrk.set_jaw(jaw1=self.psm1_jaw_opening)

		# go down toward the knot & close jaw
		if thresh:
			final_depth = min(pos1[2], -0.13411197)
		pos = [pos1[0], pos1[1], max(final_depth, self.psm1_min_depth)-0.001]
		rot = self.inclined_orientation(np.deg2rad(rot1), approach_angle)
		self.dvrk.set_pose(pos1=pos, rot1=rot)
		self.dvrk.set_jaw(jaw1=self.psm1_jaw_closing) # @PRIYA

	def grasp_psm2(self, pos2, rot2, reid=False, thresh=True, slight_grasp=False, open=True):	# using PSM2
		# go above the knot to pick up
		#pos = [pos2[0], pos2[1], self.height_ready]
		pos = [pos2[0], pos2[1], pos2[2]+self.grasp_above_offset]
		approach_angle = self.psm2_lat_rot_reid if reid else self.psm2_lat_rot
		rot = self.inclined_orientation(np.deg2rad(rot2+ self.psm2_grasp_rot_offset), approach_angle)
		if open:
			self.dvrk.set_jaw(jaw2=self.psm2_jaw_opening)
		self.dvrk.set_pose(pos2=pos, rot2=rot)
		self.dvrk.set_jaw(jaw2=self.psm2_jaw_opening)

		# go down toward the knot & close jaw
		# pos2[2] = pos2[2] if reid else pos2[2] - 0.002
		final_depth = min(pos2[2], -0.13411197) if thresh else pos2[2]
                #if thresh:
		#final_depth = final_depth -0.002 if not slight_grasp else final_depth+0.004
		final_depth = final_depth -0.002 if not slight_grasp else final_depth+0.012
		pos = [pos2[0], pos2[1], max(final_depth, self.psm2_min_depth)-0.002]
		print("Actual depth:", pos[2])
		rot = self.inclined_orientation(np.deg2rad(rot2+ self.psm2_grasp_rot_offset), approach_angle)
		self.dvrk.set_pose(pos2=pos, rot2=rot)
		if not slight_grasp:
			self.dvrk.set_jaw(jaw2=self.psm2_jaw_closing)
		#else:
		#	self.dvrk.set_jaw(jaw2=[0.3])

	def pull_psm2(self, pos2, rot2, reid=False):	# using PSM2
		# pull knot
		pos = [pos2[0], pos2[1], max(pos2[2], self.psm2_min_depth)]
		approach_angle = self.psm2_lat_rot_reid if reid else self.psm2_lat_rot
		rot = self.inclined_orientation(np.deg2rad(rot2+ self.psm2_grasp_rot_offset), approach_angle)
		self.dvrk.set_pose(pos2=pos, rot2=rot)

	def lift_psm2(self, pos2, rot2, reid=True):
		pos = [pos2[0], pos2[1], self.lift_height]
		approach_angle = self.psm2_lat_rot_reid if reid else self.psm2_lat_rot
		rot = self.inclined_orientation(np.deg2rad(rot2+ self.psm2_grasp_rot_offset), approach_angle)
		self.dvrk.set_pose(pos2=pos, rot2=rot)

	def lift_psm1(self, pos1, rot1, reid=True):
		psm1_pos = [pos1[0], pos1[1], self.lift_height]
		approach_angle = self.psm1_lat_rot_reid if reid else self.psm1_lat_rot 
		rot_psm1 = self.inclined_orientation(np.deg2rad(rot1), approach_angle)
		self.dvrk.set_pose(pos1=psm1_pos, rot1=rot_psm1)
    
	def release(self):
		self.dvrk.set_jaw(jaw1=self.psm1_jaw_opening, jaw2=self.psm2_jaw_opening)

	# ZYZ euler angle to quaternion
	def inclined_orientation(self, axis_rot, latitude, longitude=0):
	    theta_z1 = longitude
	    theta_y = latitude
	    theta_z2 = axis_rot
	    R = U.Rz(theta_z1).dot(U.Ry(theta_y)).dot(U.Rz(theta_z2))
	    return U.R_to_quaternion(R)
