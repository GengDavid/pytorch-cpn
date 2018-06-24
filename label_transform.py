import sys
sys.path.insert(0, 'cocoapi/PythonAPI')
import os
from pycocotools.coco import COCO
from tqdm import tqdm
import json
anno_root = 'data/COCO2017/annotations/'
coco_kps = COCO(os.path.join(anno_root,'person_keypoints_train2017.json'))
coco_ids = coco_kps.getImgIds()
catIds = coco_kps.getCatIds(catNms=['person'])
train_data = []
print('annotation transforming...')
for img_id in tqdm(coco_ids):
	img = coco_kps.loadImgs(img_id)[0]
	annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds)
	anns = coco_kps.loadAnns(annIds)
	for ann in anns:
		if ann['num_keypoints']==0:
			continue
		single_data = {}
		keypoints = ann['keypoints']
		bbox = ann['bbox']
		num_keypoints = ann['num_keypoints']
		file_name = img['file_name']
		unit = {}
		unit['num_keypoints'] = num_keypoints
		unit['keypoints'] = keypoints
		x1,y1,width,height = bbox
		x2 = x1+width
		y2 = y1+height
		unit['GT_bbox'] = [int(x1),int(y1),int(x2),int(y2)]
		single_data['unit'] = unit
		imgInfo = {}
		imgInfo['imgID'] = img_id
		imgInfo['img_paths'] = file_name
		single_data['imgInfo'] = imgInfo
		for i in range(4):
			tmp = single_data.copy()
			tmp['operation'] = i
			train_data.append(tmp)

print('saving transformed annotation...')
with open(os.path.join(anno_root,'COCO_2017_train.json'),'w') as wf:
    json.dump(train_data, wf)
print('done')

coco_kps = COCO(os.path.join(anno_root,'person_keypoints_val2017.json'))
coco_ids = coco_kps.getImgIds()
catIds = coco_kps.getCatIds(catNms=['person'])
train_data = []
print('annotation transforming...')
for img_id in tqdm(coco_ids):
	img = coco_kps.loadImgs(img_id)[0]
	annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds)
	anns = coco_kps.loadAnns(annIds)
	for ann in anns:
		if ann['num_keypoints']==0:
			continue
		single_data = {}
		keypoints = ann['keypoints']
		bbox = ann['bbox']
		num_keypoints = ann['num_keypoints']
		file_name = img['file_name']
		unit = {}
		unit['num_keypoints'] = num_keypoints
		unit['keypoints'] = keypoints
		x1,y1,width,height = bbox
		x2 = x1+width
		y2 = y1+height
		unit['GT_bbox'] = [int(x1),int(y1),int(x2),int(y2)]
		single_data['unit'] = unit
		imgInfo = {}
		imgInfo['imgID'] = img_id
		imgInfo['img_paths'] = file_name
		single_data['imgInfo'] = imgInfo
		for i in range(4):
			tmp = single_data.copy()
			tmp['operation'] = i
			train_data.append(tmp)

print('saving transformed annotation...')
with open(os.path.join(anno_root,'COCO_2017_val.json'),'w') as wf:
    json.dump(train_data, wf)
print('done')