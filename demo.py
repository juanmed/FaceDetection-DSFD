import cv2
import torch
import time
from face_ssd_infer import SSD
from utils import vis_detections, get_detections
device = torch.device("cuda")
conf_thresh = 0.3
target_size = (800, 800)


net = SSD("test")
net.load_state_dict(torch.load('weights/WIDERFace_DSFD_RES152.pth'))
net.to(device).eval();

img_path = '/home/fer/Videos/YDXJ2900.jpg'

img = cv2.imread(img_path, cv2.IMREAD_COLOR)

t1 = time.time()
detections = net.detect_on_image(img, target_size, device, is_pad=False, keep_thresh=conf_thresh)
t2 = time.time()
print("time: {:.2f}".format(t2-t1))

t1 = time.time()
detections = net.detect_on_image(img, target_size, device, is_pad=False, keep_thresh=conf_thresh)
t2 = time.time()
print("time: {:.2f}".format(t2-t1))

vis_detections(img, detections, conf_thresh, show_text=False)
bboxs = get_detections(detections, conf_thresh)
print(bboxs)