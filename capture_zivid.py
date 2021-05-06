from ZividCapture import ZividCapture
# import sys
# for p in sys.path:
#     if p == '/opt/ros/kinetic/lib/python2.7/dist-packages':
#         sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np

zc_overhead = ZividCapture(which_camera="overhead")
# zc_inclined = ZividCapture(which_camera="inclined")
zc_overhead.start()
#zc_inclined.start()

# while True:
#     image = zc.capture_2Dimage(color='BGR')
#     cv2.imshow("", image)
#     cv2.waitKey(1)

# check images
img_color1, img_depth1, img_point1 = zc_overhead.capture_3Dimage(color='BGR')
#mg_color2, img_depth2, img_point2 = zc_inclined.capture_3Dimage(color='BGR')
#zc_overhead.display_rgb(img_color1, block=True)
#zc_overhead.display_rgb(img_color2, block=True)
# zc.display_depthmap(img_point)
# zc.display_pointcloud(img_point, img_color)

dir = "bright"
save_depth = False

i = 1
while i < 200:
	img_color1, img_depth1, img_point1 = zc_overhead.capture_3Dimage(color='BGR')
	if save_depth:
		cv2.imwrite(dir+"/color/color%s.png"%str(i), img_color1)
		np.save(dir+'/depth/depth%s.npy'%str(i), img_depth1)
		np.save(dir+'/point/point%s.npy'%str(i), img_point1)
		# cv2.imwrite(dir+'/depth/depth%s.png'%str(i), img_depth1)
		# cv2.imwrite(dir+'/point/point%s.png'%str(i), img_point1)
	else:
		cv2.imwrite(dir+"/color%s.png"%str(i), img_color1)
	#cv2.imwrite("color_inclined.png", img_color2)
	cv2.imshow("img",img_color1)
	cv2.waitKey(0)
	i+= 1

# i = 1
# img_color1, img_depth1, img_point1 = zc_overhead.capture_3Dimage(color='BGR')
# if save_depth:
# 	cv2.imwrite(dir+"/color/color%s.png"%str(i), img_color1)
# 	cv2.imwrite(dir+'/depth/depth%s.png'%str(i), img_depth1)
# 	cv2.imwrite(dir+'/point/point%s.png'%str(i), img_point1)
# else:
# 	cv2.imwrite(dir+"/color%s.png"%str(i), img_color1)

# # cv2.imwrite(dir+"/color%s.png"%str(i), img_color1)
# #cv2.imwrite("color_inclined.png", img_color2)
# cv2.imshow("img",img_color1)
# cv2.waitKey(0)
