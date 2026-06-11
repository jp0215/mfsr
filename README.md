# MFSR: Multi-fractal Feature for Super-resolution Reconstruction with Fine Details Recovery

## Brief

This is an implementation of MFSR by PyTorch.Thank you for your reading!
Once the paper is accepted, we will refine the code and release it as soon as possible.

## Usage

### Environment

```shell
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

### Data Prepare

Download the dataset and prepare it in **LMDB** or **PNG** format using script.

```shell
python data/prepare_data.py  --path [data_path]  --out [result_path] --size 40,160
```

The obtained data is organized as follows:

```
# set the high/low resolution images, bicubic interpolation images path 
dataset/celebahq_16_128/
├── hr_128 # it's same with sr_16_128 directory if you don't have ground-truth images.
├── lr_16 # vinilla low resolution images
└── sr_16_128 # images ready to super resolution
```

Then change the dataset config to your data path and image resolution: 

```json
"datasets" : {
    "train": {
        "dataroot": "[output root] in prepare.py script",
        "l_resolution": "low resolution need to super_resolution",
        "r_resolution": "high resolution",
        "datatype": "lmdb or img, path of img files"
    },
    "val": {
        "dataroot": "[output root] in prepare.py script"
    }
},
```

### Pre-train CNN and generate predicted images

Modify the parameters in several files in the /pretrain_CNN directory, and then run the following script directly.

```shell
python ./pretrain_CNN/train.py
```

```shell
nohup python ./pretrain_CNN/train.py > ffhq_train.log &
```

The CNN predictions will be written to the specified path, 
note that the path needs to be specified as the previously generated **dataset/xxx/sr_[lr]_[hr]**.

### Training/Resume Training

```shell
python sr.py -p train -c [config file]
```

# 修改学习率

### Test/Evaluation

```shell
# Edit json to add pretrain model path and run the evaluation 
python sr.py -p val -c [config file]

# Quantitative evaluation alone using SSIM/PSNR metrics on given result root
python eval.py -p [result root]
```

### Inference Alone

Set the  image path like steps in `Own Data`, then run the script:

```shell
# run the script
python infer.py -c [config file]

```
## Acknowledgement
Our code is built upon the open-source project https://github.com/LYL1015/ResDiff.







