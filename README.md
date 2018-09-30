# PyTorch CPN(Cascaded Pyramid Network)

This is a PyTorch re-implementation of CPN ([Cascaded Pyramid Network](https://arxiv.org/abs/1711.07319)), winner of MSCOCO keypoints2017 challenge. The TensorFlow version can be found [here](https://github.com/chenyilun95/tf-cpn), which is implemented by the paper author.

## Evaluation results on COCO minival dataset
<center>

| Method | Base Model | Input Size | BBox | AP @0.5:0.95 | AP @0.5 | AP @0.75 | AP medium | AP large |
|:-------|:--------:|:-----:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| CPN | ResNet-50 | 256x192 | Ground Truth | 71.2 | 91.4 | 78.3 | 68.6 | 75.2 |
| CPN | ResNet-50 | 256x192 | Detection Result | 69.2 | 88.0 | 76.2 | 65.8 | 75.6 |
| CPN | ResNet-50 | 384x288 | Ground Truth | 74.1 | 92.5 | 80.6 | 70.6 | 79.5 |
| CPN | ResNet-50 | 384x288 | Detection Result | 72.2 | 89.2 | 78.6 | 68.1 | 79.3 |
| CPN | ResNet-101<sup>*</sup> | 384x288 | Ground Truth | 74.0 | 92.3 | 80.6 | 71.1 | 78.7 |
| CPN | ResNet-101<sup>*</sup> | 384x288 | Detection Result | 72.3 | 89.2 | 78.9 | 68.7 | 79.1 |
</center>

Thanks [Tiamo666](https://github.com/Tiamo666) and [mingloo](https://github.com/mingloo) for training and testing ```ResNet-50-384x288CPN``` model. And thanks [Tiamo666](https://github.com/Tiamo666) for training and testing ```ResNet-101-384x288CPN``` model.  
If you have interests in this repo, welcome to test other model configurations together.  

\* CPN-ResNet-101-384x288 model is fine-tuned from the previous pre-trained model. If you train it from scratch, it should get a higher result.

## Usage

### For training
1. Clone the repository
```
git clone https://github.com/GengDavid/pytorch-cpn
```

We'll call the directory that you cloned ```ROOT_DIR```.

2. Download MSCOCO2017 images and annotations from [http://cocodataset.org/#download](http://cocodataset.org/#download). And put images and annotation files follow the struture showed in [data/README.md](https://github.com/GengDavid/pytorch-cpn/blob/master/data/README.md)  
After placing data and annotation files. Please run ```label_transform.py``` at ```ROOT_DIR``` to transform the annotation fomat.

3. Initialize cocoapi
```
git submodule init
git submodule update
cd cocoapi/PythonAPI
make
```
It will build cocoapi tools automatically.

4. Install requirement  
  This repo require following dependences.
  - **PyTorch >= 0.4.1**
  - numpy >= 1.7.1
  - scipy >= 0.13.2
  - python-opencv >= 3.3.1
  - tqdm > 4.11.1
  - skimage >= 0.13.1

5. Training
```
cd ROOT_DIR/MODEL_DIR/
python3 train.py
```

For example, to train CPN with input resolution 256x192, just change directory into ROOT_DIR/256.192.model, and run the script.

For more args, see by using
```
python train.py --help
```

### For Validation
```
cd ROOT_DIR/MODEL_DIR/
python3 test.py -t PRE-TRAINED_MODEL_NAME
```

```-t``` meas use which pre-trained model to test.   
For more args, see by using
```
python test.py --help
```

If you want to test a pre-trained model, please place the pre-trained model into ```ROOT_DIR/MODEL_DIR/checkpoint``` directory. Please make sure your have put the corresponding model into the folder.

For example, to run pre-trained CPN model with input resolution 256x192,
```
python3 test.py -t 'CPN256x192'
```

This pre-trained model is provided below.

## Pre-trained models:
[COCO.res50.256x192.CPN](https://drive.google.com/open?id=1pUpU8o6QtgK197vAfCtT5cxokE9p-yuB) (**updated!**)  
[COCO.res50.384x288.CPN](https://drive.google.com/open?id=1L6Qq-incr2XtptdLJti3Zdf6sr19zs7y) (**updated!**)  
[COCO.res101.384x288.CPN<sup>*</sup>](https://drive.google.com/open?id=1zFYUsDbFG3xxMoZkSv253QsVD9AF9rv7) (**new**)  
\* CPN-ResNet-101-384x288 model is fine-tuned from the previous pre-trained model. If you train it from scratch, it should get a higher result.  

## Detection results on Minival dataset
The detection results are tranformed from results in [tf version](https://github.com/chenyilun95/tf-cpn) of cpn.  
[detection_minival](https://drive.google.com/open?id=1Iv6mH9DC0ia5POBFjI_MFWO2viG53TKA)

## Acknowledgements
Thanks [chenyilun95](https://github.com/chenyilun95), [bearpaw](https://github.com/bearpaw) and [last-one](https://github.com/last-one) for sharing their codes, which helps me a lot to build this repo.  
Thanks [Tiamo666](https://github.com/Tiamo666) for testing ```ResNet-50-384x288CPN``` and ```ResNet-101-384x288CPN```model.   
Thanks [mingloo](https://github.com/mingloo) for contribution.  
Thanks [mkocabas](https://github.com/mkocabas) for helping me test other configurations.  

## Others
If you have any questions or find some mistakes about this re-implementation, please open an [issue](https://github.com/GengDavid/pytorch-cpn/issues) to let me know.  
If you want to know more details about the original implementation, you can check [tf version](https://github.com/chenyilun95/tf-cpn) of cpn.

## Troubleshooting
1. Thanks [Tiamo666](https://github.com/Tiamo666) to point it out that the refineNet is implemented in a different way from the original paper(this can reach a higher results, but it will cost more memory).  
2. See issue [#10](https://github.com/GengDavid/pytorch-cpn/issues/10) and issue [#7](https://github.com/GengDavid/pytorch-cpn/issues/7).  
Codes and results have been updated!(2018/9/6)

## Reference
[1] Chen, Y., Wang, Z., Peng, Y., Zhang, Z., Yu, G., Sun, J.: Cascaded pyramid network for multi-person pose estimation. CVPR (2018)
