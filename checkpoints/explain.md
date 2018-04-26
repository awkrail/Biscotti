# checkpoints
In this directory, you can see hdf5 models. 
After training with keras, execute hdf5to_pb.py.
It convert hdf5 into pb, which is available to tensorflow.

## 3layer
In this directory, there are trained model which train ycbcr images and DCT coefficient after guetzli

## 3layer_rgb
In this directory, there are trained model which train rgb images and DCT coefficient after guetzli

## 3layer_rgb_model
In this directory, there are trained model which train rgb images and DCT coefficient after guetzli
These models are saved keras.models.save() because tensorflow models need this method when you convert your hdf5 model into pb model.

## generator
In this directory, there are trained model which train ycbcr images and DCT coefficient after guetzli.
In this points, it is the same 3layer, but in this directory, we use pix2pix model.
