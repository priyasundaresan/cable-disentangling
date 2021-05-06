import cv2
import numpy as np
import os
import os.path as osp

class PixelSelector:
    def __init__(self):
        pass

    def load_image(self, img, recrop=False):
        self.img = img
        if recrop:
            cropped_img = self.crop_at_point(img, 700, 300, width=400, height=300)
            self.img = cv2.resize(cropped_img, (640, 480))

    def crop_at_point(self, img, x, y, width=640, height=480):
        img = img[y:y+height, x:x+width]
        return img

    def mouse_callback(self, event, x, y, flags, param):
        print(self.img)
        print("HERE")
        cv2.imshow("pixel_selector", self.img)
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.clicks.append([x, y])
            print(x, y)
            cv2.circle(self.img, (x, y), 3, (255, 255, 0), -1)

    def run(self, img):
        self.load_image(img)
        self.clicks = []
        cv2.namedWindow('pixel_selector')
        cv2.setMouseCallback('pixel_selector', self.mouse_callback)
        # print("HI", self.img)
        while True:
            # cv2.imshow('pixel_selector',self.img)
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
        print(self.clicks)
        return self.clicks

exp_dir = "../nonplanar/11"

if __name__ == '__main__':
    pixel_selector = PixelSelector()
    pixel_selector.run(cv2.imread('./test.png'))
    #for file in os.listdir(os.path.join(exp_dir, "all_actions")):
    #    print(file)
    #    if not "segdepth" in file or not ".png" in file:
    #        continue

    #    img = cv2.imread(osp.join(exp_dir, "all_actions", file))
    #    points = np.array(pixel_selector.run(img))
    #    assert len(points) > 0
    #    np.save(osp.join(exp_dir, "all_actions", file[:-4] + ".npy"), points)

    #for file in os.listdir(os.path.join(exp_dir, "images/phoxi")):
    #    if not "segdepth" in file or not ".png" in file:
    #        continue
    #    print(file)
    #    img = cv2.imread(osp.join(exp_dir, "images/phoxi", file))
    #    points = np.array(pixel_selector.run(img))
    #    assert len(points) > 0
    #    np.save(osp.join(exp_dir, "images/phoxi", file[:-4] + ".npy"), points)
