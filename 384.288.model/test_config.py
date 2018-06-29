import os
import os.path
import sys
import numpy as np

def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)
        
class Config:
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    this_dir_name = cur_dir.split('/')[-1]
    root_dir = os.path.join(cur_dir, '..')

    model = 'CPN50' # option 'CPN50', 'CPN101'

    num_class = 17
    img_path = os.path.join(root_dir, 'data', 'COCO2017', 'val2017')
    symmetry = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
    bbox_extend_factor = (0.1, 0.15) # x, y

    pixel_means = np.array([122.7717, 115.9465, 102.9801]) # RGB
    data_shape = (384, 288) #height, width
    output_shape = (96, 72) #height, width

    use_GT_bbox = True
    if use_GT_bbox:
        gt_path = os.path.join(root_dir, 'data', 'COCO2017', 'annotations', 'COCO_2017_val.json')
    else:
        # if False, make sure you have downloaded the val_dets.json and place it into annotation folder
        gt_path = os.path.join(root_dir, 'data', 'COCO2017', 'annotations', 'val_dets.json')
        
    ori_gt_path = os.path.join(root_dir, 'data', 'COCO2017', 'annotations', 'person_keypoints_val2017.json')

cfg = Config()
add_pypath(cfg.root_dir)
add_pypath(os.path.join(cfg.root_dir, 'cocoapi/PythonAPI'))
