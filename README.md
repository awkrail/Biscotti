# Biscotti
[WIP]Faster Implement than Guetzli with Deep Learning

# Requirements
Python >= 3.0
Keras >= 2.0
Tensorflow == 1.7.0
numpy
opencv
pandas
matplotlib

run `pip install requirements.txt`

# Usage
WIP

# Train
## 1. Make Dataset
You should set your images(jpg) at `images/`

After that, run `./train_bin/script/dump_qopt_images.sh`

This script will make necessary folders

**[WATCH]** This script is very slow because of guetzli

## 2. Training
If you want to train unet, run `python src/train_unet.py [options]`

There are 5 options.
1. datasetpath .. training dataset path. if you run `./train_bin/script/dump_qopt_images.sh`, write `train/`
2. outputfile .. keras output model. I recommend `checkpoints/` directory.
3. batch_size .. batchsize is the number of training examples utilised in one iteration.(default 32)
4. epoch .. one pass of the full training set.(default 400)

if yo want to train pix2pix, it is added patch_size option.(I adopt PatchGAN(https://arxiv.org/pdf/1611.07004.pdf))

5. patch_size .. how many divided patches(default 112 because I adopt 224*224 size images when training)

## 3. evaluation
Please run `train_bin/Release/guetzli_dumper`. It will dump image coefficient after guetzli, and save csv/

Please run `python src/predict.py`

There are 8 options.

1. modelpath .. the file where you save your keras model.
2. imagepath .. image you want to evaluate.
3. targetsize .. image size you want to evaluate.(default 224) 
4. resultpath .. path which save results
5. csvpath .. path which save result csv
6. guetzli_csv_path .. path which 

**[TODO]** Now there are some bug unless you input images which can be divided 16.

**[TODO]** I will change method to evaluate. This method is not good.

# Support
- YUV420

# Not Support
- YUV444
- png
- GrayScale
