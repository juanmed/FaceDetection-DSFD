## [DSFD: Dual Shot Face Detector](https://arxiv.org/abs/1810.10220)

By [Jian Li](https://lijiannuist.github.io/), [Yabiao Wang](https://github.com/ChaunceyWang), [Changan Wang](https://github.com/HiKapok), [Ying Tai](https://tyshiwo.github.io/), [Jianjun Qian](http://www.escience.cn/people/JianjunQian/index.html), [Jian Yang](https://scholar.google.com/citations?user=6CIDtZQAAAAJ&hl=zh-CN&oi=sra), Chengjie Wang, Jilin Li, Feiyue Huang.


### Simple test on image

```python
import cv2
import torch
from face_ssd_infer import SSD
from utils import vis_detections


device = torch.device("cpu")
conf_thresh = 0.3
target_size = (800, 800)


net = SSD("test")
net.load_state_dict(torch.load('weights/WIDERFace_DSFD_RES152.pth'))
net.to(device).eval();

img_path = './imgs/11_Meeting_Meeting_11_Meeting_Meeting_11_304.jpg'

img = cv2.imread(img_path, cv2.IMREAD_COLOR)
detections = net.detect_on_image(img, target_size, device, is_pad=False, keep_thresh=conf_thresh)
vis_detections(img, detections, conf_thresh, show_text=False)

```


### Requirements

- Torch >= 1.0.0
- Torchvision >= 0.2.1
- Numpy >=  1.14.2
- opencv-python >= 4.0
- Matplotlib


### Getting Started

```bash
git clone https://github.com/vlad3996/FaceDetection-DSFD.git
cd FaceDetection-DSFD
pip install -r requirements.txt

python demo.py

```



### Citation
If you find DSFD useful in your research, please consider citing: 
```
@inproceedings{li2018dsfd,
  title={DSFD: Dual Shot Face Detector},
  author={Li, Jian and Wang, Yabiao and Wang, Changan and Tai, Ying and Qian, Jianjun and Yang, Jian and Wang, Chengjie and Li, Jilin and Huang, Feiyue},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```
