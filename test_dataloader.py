from mscocoMulti import MscocoMulti
import torch
import torch.nn as nn
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.datasets as datasets
import skimage.io
import cv2
import os
import os
import argparse
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

class Pool(nn.Module):
    def __init__(self):
        super(Pool, self).__init__()

        self.p = nn.MaxPool2d(kernel_size = 17, stride=1)

    def forward(self, x):
        # out = F.adaptive_max_pool2d(x, 64)
        # out = self.p(x)
        out = F.upsample(x, size=(64, 64), mode='bilinear')

        return out
VISIBLE = False
symmetry = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
mean = torch.Tensor([122.7717, 102.9801, 115.9465])
IMAGE_DIR = '/home/zgw/Documents/L2G_PoseEstimation/data/COCO2017/train2017'
if __name__ == '__main__': 
    train_loader = torch.utils.data.DataLoader(
        MscocoMulti('new_local_label.json', '/home/zgw/Documents/L2G_PoseEstimation/data/COCO2017/train2017',
        symmetry, mean),
        batch_size=1, shuffle=False,
        num_workers=1, pin_memory=True)  
    size = 'small'
    pool = Pool()

    for i, (inputs, targets, valid, meta) in enumerate(train_loader):
        if i%100 == 0:
            print(i)

        import pylab

        ids = meta['imgID'].numpy()
        bbox = meta['GT_bbox'].numpy()
        score_map = targets[-4].numpy()
        for b in range(1):
            details = meta['augmentation_details']
            if VISIBLE == True:
                path = meta['img_path'][b]
                image = skimage.io.imread(path)
                img_name = path.split('/')[-1].split('_')[0]
                ori_globalimage = skimage.io.imread(os.path.join(IMAGE_DIR,img_name))
                fig = plt.figure()
                for p in range(17):
                    ax = fig.add_subplot(3,6,p+1)
                    ax.imshow(score_map[b][p])     
                    print(np.max(score_map[b][p]))
                # fig2 = plt.figure()
                # for p in range(17):
                #     ax = fig2.add_subplot(3,6,p+1)
                #     ax.imshow(target[b][p])        
                # fig3 = plt.figure()
                # # for p in range(6):
                   # #  ax = fig3.add_subplot(2,3,p+1)
                   # #  ax.imshow(batch_data['ori_localimage'][p])       
                # ax = fig3.add_subplot(111)
                # ax.imshow(image) 

                fig4 = plt.figure()
                ax = fig4.add_subplot(111)
                ax.imshow(ori_globalimage)  


            single_result_dict = {}
            single_result = []
            x1, y1, x2, y2 = bbox[b]
            height = y2 - y1
            width = x2 - x1
            
            single_map = score_map[b]
            r0 = single_map.copy()
            r0 /= 255
            r0 += 0.5
            v_score = np.zeros(17)
            for p in range(17): 
                single_map[p] /= np.amax(single_map[p])
                border = 10
                dr = np.zeros((64 + 2*border, 48+2*border))
                dr[border:-border, border:-border] = single_map[p].copy()
                dr = cv2.GaussianBlur(dr, (21, 21), 0)

                max_val = np.max(dr)
                if(max_val<0.1):
                    x = pmean_dict['mean'][p][0]
                    y = pmean_dict['mean'][p][1]
                    px = x
                    py = y    
                    print('-----------------')           
                else:
                    lb = dr.argmax()
                    y, x = np.unravel_index(lb, dr.shape)
                    dr[y, x] = 0
                    lb = dr.argmax()
                    py, px = np.unravel_index(lb, dr.shape)
                    y -= border
                    x -= border
                    py -= border + y
                    px -= border + x
                    ln = (px ** 2 + py ** 2) ** 0.5
                    delta = 0.25
                    if ln > 1e-3:
                        x += delta * px / ln
                        y += delta * py / ln
                    x = max(0, min(x, 48 - 1))
                    y = max(0, min(y, 64 - 1))
                resx = (4 * x + 2) / 192 * (details[b][2] - details[b][0]) + details[b][0]
                resy = (4 * y + 2) / 256 * (details[b][3] - details[b][1]) + details[b][1]
                v_score[p] = float(r0[p, int(round(y)+1e-10), int(round(x)+1e-10)])
                # print(p_max)
                if VISIBLE == True:
                    print(resx, resy)
                    pylab.plot(resx,resy,'r*')                       
                single_result.append(resx)
                single_result.append(resy)
                single_result.append(1)   
            pylab.show()