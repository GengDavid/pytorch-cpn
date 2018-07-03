# PyTorch CPN(Cascaded Pyramid Network)

This is a PyTorch re-implementation of CPN ([Cascaded Pyramid Network](https://arxiv.org/abs/1711.07319)). The TensorFlow version can be found [here](https://github.com/chenyilun95/tf-cpn), which is implemented by the paper author.

## Evaluation results on COCO minival dataset
<center>

| Method | Base Model | Input Size | BBox | AP @0.5:0.95 | AP @0.5 | AP @0.75 | AP medium | AP large |
|:-------|:--------:|:-----:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| CPN | ResNet-50 | 256x192 | Ground Truth |71.0 | 90.4 | 78.2 | 68.4 | 75.3 |
| CPN | ResNet-50 | 256x192 | Detection Result |69.9 | 88.2 | 76.8 | 66.4 | 76.5 |

</center>

I only have tested ResNet-50-256x192 model because I don't have enough GPUs to test the models. If you have interests in this repo, welcome to test other model configurations together.  

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
  - PyTorch == 0.4.0
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
python train.py --help
```

If you want to test a pre-trained model, please place the pre-trained model into ```ROOT_DIR/MODEL_DIR/checkpoint``` directory. Please make sure your have put the corresponding model into the folder.

For example, to run pre-trained CPN model with input resolution 256x192,
```
python3 mptest.py -t 'CPN256x192'
```

This pre-trained model is provided below.

## Pre-trained models:
[COCO.res50.256x192.CPN](https://drive.google.com/open?id=1K16rW53JKa99hxpvAZ8_JHzZcBPFls_D)

## Detection results on Minival dataset
The detection results are tranformed from results in [tf version](https://github.com/chenyilun95/tf-cpn) of cpn.  
[detection_minival](https://drive.google.com/open?id=1Iv6mH9DC0ia5POBFjI_MFWO2viG53TKA)

## Acknowledgements
Thanks [chenyilun95](https://github.com/chenyilun95), [bearpaw](https://github.com/bearpaw) and [last-one](https://github.com/last-one) for sharing their codes, which helps me a lot to build this repo.

## Others
If you have any questions or find some mistakes about this re-implementation, please open an [issue](https://github.com/GengDavid/pytorch-cpn/issues) to let me know.  
If you want to know more details about the original implementation, you can check [tf version](https://github.com/chenyilun95/tf-cpn) of cpn.

