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

    model = 'CPN50'

    lr = 5e-4
    lr_gamma = 0.5
    lr_dec_epoch = list(range(6,40,6))

    batch_size = 32
    weight_decay = 1e-5

    num_class = 17
    img_path = os.path.join(root_dir, 'data', 'COCO2017', 'train2017')
    symmetry = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
    bbox_extend_factor = (0.1, 0.15) # x, y

    # data augmentation setting
    scale_factor=(0.7, 1.35)
    rot_factor=45

    pixel_means = np.array([122.7717, 115.9465, 102.9801]) # RGB
    data_shape = (256, 192)
    output_shape = (64, 48)
    gaussain_kernel = (7, 7)
    
    gk15 = (15, 15)
    gk11 = (11, 11)
    gk9 = (9, 9)
    gk7 = (7, 7)

    gt_path = os.path.join(root_dir, 'data', 'COCO2017', 'annotations', 'COCO_2017_train.json')

cfg = Config()
add_pypath(cfg.root_dir)

