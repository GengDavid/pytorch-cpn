import os
import os.path
import sys
import numpy as np

class Config:
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    this_dir_name = cur_dir.split('/')[-1]

    model = 'CPN50' # option 'CPN50', 'CPN101'

    num_class = 17
    # img_path = os.path.join(cur_dir, 'data', 'COCO2017', 'val2017')
    img_path = '/home/zgw/Documents/L2G_PoseEstimation/data/COCO2017/val2017'
    symmetry = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]

    pixel_means = np.array([122.7717, 115.9465, 102.9801]) # RGB
    data_shape = (256, 192)
    output_shape = (64, 48)

    gt_path = os.path.join(cur_dir, 'data', 'COCO2017', 'annotations', 'COCO_2017_val.json')
    ori_gt_path = os.path.join(cur_dir, 'data', 'COCO2017', 'annotations', 'person_keypoints_val2017.json')

cfg = Config()

