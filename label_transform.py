import sys
sys.path.insert(0, 'cocoapi/PythonAPI')
import os
from pycocotools.coco import COCO
from tqdm import tqdm
import json
from utils.osutils import isfile

anno_root = 'data/COCO2017/annotations/'

def trans_anno(ori_file, target_file, is_val):
	file_exist=False
	no_ori=False
	train_anno = os.path.join(anno_root, target_file)
	if isfile(train_anno):
		file_exist = True
	ori_anno = os.path.join(anno_root,ori_file)
	if isfile(ori_anno)==False:
		no_ori = True
	if file_exist==False and no_ori==False:
		coco_kps = COCO(ori_anno)
		coco_ids = coco_kps.getImgIds()
		catIds = coco_kps.getCatIds(catNms=['person'])
		train_data = []
		print('transforming annotations...')
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
				if is_val==False:
					for i in range(4):
						tmp = single_data.copy()
						tmp['operation'] = i
						train_data.append(tmp)
				else:
					single_data['score'] = 1
					train_data.append(single_data)
		print('saving transformed annotation...')
		with open(train_anno,'w') as wf:
		    json.dump(train_data, wf)
		print('done')
	if no_ori:
		print('''WARNING! There is no annotation file find at {}. 
			Make sure you have put annotation files into the right folder.'''
			.format(ori_anno))

trans_anno('person_keypoints_train2017.json', 'COCO_2017_train.json', False)
trans_anno('person_keypoints_val2017.json', 'COCO_2017_val.json', True)
