# PyTorch CPN(Cascaded Pyramid Network)

This is a PyTorch re-implementation of CPN ([Cascaded Pyramid Network](https://arxiv.org/abs/1711.07319)). The TensorFlow version can be found [here](https://github.com/chenyilun95/tf-cpn), which is implemented by the paper author.

## Results on COCO minival dataset
<center>

| Method | Base Model | Input Size | AP @0.5:0.95 | AP @0.5 | AP @0.75 | AP medium | AP large |
|:-------|:--------:|:-----:|:-------:|:-------:|:-------:|:-------:|:-------:|
| CPN | ResNet-50 | 256x192 | x | x | x | x | x |

</center>


## Usage

### For training
1. Clone the repository
```
git clone https://github.com/GengDavid/pytorch-cpn
```


2. Download MSCOCO images from [http://cocodataset.org/#download](http://cocodataset.org/#download). And put images and annotation files follow the struture showed in data/README.md

3. Initialize COCOapi
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

5. Training.
```
cd %ROOT_DIR%/%MODEL_DIR%/
python3 train.py
```

### For Validation
```
cd %ROOT_DIR%/%MODEL_DIR%/
python3 test.py -t PRE-TRAINED_MODEL_NAME
```

For example
```
python3 mptest.py -t 'epoch40checkpoint'
```

### Pre-trained models:

[COCO.res50.256x192.CPN]()


## Others
If you have any questions or find some mistakes about this re-implementation, please open an issue to let me know.  
If you want to know more details about the original implementation, you can check tf version of cpn.

