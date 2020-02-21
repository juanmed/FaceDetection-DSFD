import cv2
import torch
import time
from face_ssd_infer import SSD
from utils import get_detections


class DSFD_detector():

    def __init__(self, conf_thresh = 0.3, target_size = (800, 800), device = 'cuda'):

        self.device = torch.device(device)
        self.conf_thresh = conf_thresh
        self.target_size = (800, 800)
        self.net = SSD("test")
        self.net.load_state_dict(torch.load('weights/WIDERFace_DSFD_RES152.pth'))
        self.net.to(self.device)

    def detect(self, image):
        detections = self.net.detect_on_image(image, self.target_size, self.device, is_pad=False, keep_thresh=self.conf_thresh)
        bboxs = get_detections(detections, self.conf_thresh)
        return [0,bboxs,0]

if __name__ == '__main__':
    
    img_path = '/home/fer/Videos/YDXJ2900.jpg'

    detector = DSFD_detector()
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    bboxs = detector.detect(img)
    print(bboxs)
