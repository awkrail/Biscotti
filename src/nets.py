import numpy as np

from keras.layers.core import Flatten, Dense, Activation, Lambda
from keras.models import Model
from keras.layers.convolutional import Conv2D, Deconv2D, ZeroPadding2D, UpSampling2D
from keras.layers import Input, Concatenate, concatenate, LeakyReLU, BatchNormalization

# PatchGAN-discriminator
def discriminator(img_shape, disc_img_shape, patch_num):
    disc_raw_img_shape = (disc_img_shape[0], disc_img_shape[1], img_shape[-1])
    list_input = [Input(shape=disc_img_shape, name='disc_input_'+str(i)) for i in range(patch_num)] # => DCT
    list_raw_input = [Input(shape=disc_raw_img_shape, name='disc_raw_input_'+str(i)) for i in range(patch_num)] # => Input Image
    
    filter_num = 64
    conv_num = int(np.floor(np.log(disc_img_shape[1]) / np.log(2)))
    list_filters = [filter_num*min(8, (2**i)) for i in range(conv_num)]

    # First_Convolution_for_generated_images
    generated_patch_input = Input(shape=disc_img_shape, name="discriminator_dct_input") # DCT
    xg = Conv2D(list_filters[0], kernel_size=(3, 3), strides=(2, 2), name="disc_conv2d_1", padding="same")(generated_patch_input)
    xg = BatchNormalization(axis=-1)(xg)
    xg = LeakyReLU(0.2)(xg)

    # first_Convolution_for_predicted_guetzli_DCT
    raw_patch_input = Input(shape=disc_raw_img_shape, name="discriminator_image_input") # Raw Input
    xr = Conv2D(list_filters[0], kernel_size=(3, 3), strides=(2, 2), name="dic_dct_conv2d_1", padding="same")(raw_patch_input)
    xr = BatchNormalization(axis=-1)(xr)
    xr = LeakyReLU(0.2)(xr)

    # Next Conv
    for i, f in enumerate(list_filters[1:]):
        x = Concatenate(axis=-1)([xg, xr])
        x = Conv2D(f, (3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization(axis=-1)(x)
        x = LeakyReLU(0.2)(x)
    
    x_flat = Flatten()(x)
    x = Dense(2, activation='softmax', name='discriminator_dense')(x_flat)

    PatchGAN = Model(inputs=[generated_patch_input, raw_patch_input], outputs=[x], name='PatchGAN')
    print('PatchGAN Summary')
    PatchGAN.summary()
    
    x = [PatchGAN([list_input[i], list_raw_input[i]]) for i in range(patch_num)]

    if len(x) > 1:
        x = Concatenate(axis=-1)(x)
    else:
        x = x[0]
    
    x_out = Dense(2, activation='softmax', name='discriminator_output')(x)

    discriminator_model = Model(inputs=(list_input+list_raw_input), outputs=[x_out], name="Discriminator")
    return discriminator_model

def generator_3layer(input_size):
    inputs = Input((input_size[0], input_size[1], 3))
    conv1 = Conv2D(32, (3, 3), padding='same')(inputs)
    conv1 = LeakyReLU(0.2)(conv1)

    conv2 = Conv2D(64, (3, 3), strides=(2, 2), padding='same')(conv1)
    conv2 = LeakyReLU(0.2)(conv2)
    conv2 = BatchNormalization(axis=-1)(conv2)

    conv3 = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(conv2)
    conv3 = LeakyReLU(0.2)(conv3)
    conv3 = BatchNormalization(axis=-1)(conv3)

    conv4 = Conv2D(256, (3, 3), strides=(2, 2), padding='same')(conv3)
    conv4 = Conv2D(256, (3, 3), padding="same")(conv4)
    conv4 = Activation('relu')(conv4)

    # Decoder Side
    up1 = concatenate([UpSampling2D(size=(2, 2), data_format="channels_last")(conv4), conv3], axis=3)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(up1)
    conv5 = BatchNormalization(axis=-1)(conv5)

    up2 = concatenate([UpSampling2D(size=(2, 2), data_format="channels_last")(conv5), conv2], axis=3)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(up2)
    conv6 = BatchNormalization(axis=-1)(conv6)
    
    up3 = concatenate([UpSampling2D(size=(2, 2), data_format="channels_last")(conv6), conv1], axis=3)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up3)
    conv7 = BatchNormalization(axis=-1)(conv7)

    conv8 = Conv2D(3, (1, 1), activation='sigmoid', data_format="channels_last")(conv7)
    fcn = Model(input=inputs, output=conv8)

    return fcn


# U-Net
def generator(input_size):
    # EncoderSide
    inputs = Input((input_size[0], input_size[1], 3))

    conv1 = Conv2D(32, (3, 3), padding='same', data_format="channels_last")(inputs)
    conv1 = LeakyReLU(0.2)(conv1)

    conv2 = Conv2D(64, (3, 3), strides=(2, 2), padding='same', data_format="channels_last")(conv1)
    conv2 = LeakyReLU(0.2)(conv2)
    conv2 = BatchNormalization(axis=-1)(conv2)

    conv3 = Conv2D(128, (3, 3), strides=(2, 2), padding='same', data_format="channels_last")(conv2)
    conv3 = LeakyReLU(0.2)(conv3)
    conv3 = BatchNormalization(axis=-1)(conv3)

    conv4 = Conv2D(256, (3, 3), strides=(2, 2), padding='same', data_format="channels_last")(conv3)
    conv4 = LeakyReLU(0.2)(conv4)
    conv4 = BatchNormalization(axis=-1)(conv4)

    conv5 = Conv2D(512, (3, 3), strides=(2, 2), padding='same', data_format="channels_last")(conv4)
    conv5 = LeakyReLU(0.2)(conv5)
    conv5 = BatchNormalization(axis=-1)(conv5)

    conv6 = Conv2D(1024, (3, 3), strides=(2, 2), padding='same', data_format="channels_last")(conv5)
    conv6 = Conv2D(1024, (3, 3), padding='same', data_format="channels_last")(conv6)
    conv6 = Activation('relu')(conv6)

    # Decoder Side
    up1 = concatenate([UpSampling2D(size=(2, 2), data_format="channels_last")(conv6), conv5], axis=3)
    conv7 = Conv2D(512, (3, 3), activation='relu', padding='same', data_format="channels_last")(up1)
    conv7 = BatchNormalization(axis=-1)(conv7)

    up2 = concatenate([UpSampling2D(size=(2, 2), data_format="channels_last")(conv7), conv4], axis=3)
    conv8 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format="channels_last")(up2)
    conv8 = BatchNormalization(axis=-1)(conv8)

    up3 = concatenate([UpSampling2D(size=(2, 2), data_format="channels_last")(conv8), conv3], axis=3)
    conv9 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_last")(up3)
    conv9 = BatchNormalization(axis=-1)(conv9)

    up4 = concatenate([UpSampling2D(size=(2, 2), data_format="channels_last")(conv9), conv2], axis=3)
    conv10 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_last")(up4)
    conv10 = BatchNormalization(axis=-1)(conv10)

    up5 = concatenate([UpSampling2D(size=(2, 2), data_format="channels_last")(conv10), conv1], axis=3)
    conv11 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format="channels_last")(up5)
    conv11 = BatchNormalization(axis=-1)(conv11)

    conv12 = Conv2D(3, (1, 1), activation='sigmoid', data_format="channels_last")(conv11)
    fcn = Model(input=inputs, output=conv12)

    return fcn

def generator_discriminator(generator, discriminator, img_shape, patch_size):
    raw_input = Input(shape=img_shape, name='DCGAN_input')
    generated_image = generator(raw_input)

    height, width = img_shape[:-1]
    pheight, pwidth = patch_size, patch_size

    # split input patchsize
    list_row_idx = [(i*pheight, (i+1)*pheight) for i in range(height//pheight)]
    list_col_idx = [(i*pwidth, (i+1)*pwidth) for i in range(width//pwidth)]

    list_gen_patch = []
    list_raw_patch = []
    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            raw_patch = Lambda(lambda z: z[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])(raw_input)
            list_raw_patch.append(raw_patch)
            x_patch = Lambda(lambda z: z[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])(generated_image)
            list_gen_patch.append(x_patch)
    
    DCGAN_output = discriminator(list_gen_patch+list_raw_patch)

    DCGAN = Model(inputs=[raw_input],
                  outputs=[generated_image, DCGAN_output],
                  name="DCGAN")
    return DCGAN

def generator_butteraugli(input_size, model_path):
    # EncoderSide
    inputs = Input((input_size[0], input_size[1], 3))

    conv1 = Conv2D(32, (3, 3), padding='same', data_format="channels_last")(inputs)
    conv1 = LeakyReLU(0.2)(conv1)

    conv2 = Conv2D(64, (3, 3), strides=(2, 2), padding='same', data_format="channels_last")(conv1)
    conv2 = LeakyReLU(0.2)(conv2)
    conv2 = BatchNormalization(axis=-1)(conv2)

    conv3 = Conv2D(128, (3, 3), strides=(2, 2), padding='same', data_format="channels_last")(conv2)
    conv3 = LeakyReLU(0.2)(conv3)
    conv3 = BatchNormalization(axis=-1)(conv3)

    conv4 = Conv2D(256, (3, 3), strides=(2, 2), padding='same', data_format="channels_last")(conv3)
    conv4 = LeakyReLU(0.2)(conv4)
    conv4 = BatchNormalization(axis=-1)(conv4)

    conv5 = Conv2D(512, (3, 3), strides=(2, 2), padding='same', data_format="channels_last")(conv4)
    conv5 = LeakyReLU(0.2)(conv5)
    conv5 = BatchNormalization(axis=-1)(conv5)

    conv6 = Conv2D(1024, (3, 3), strides=(2, 2), padding='same', data_format="channels_last")(conv5)
    conv6 = Conv2D(1024, (3, 3), padding='same', data_format="channels_last")(conv6)
    conv6 = Activation('relu')(conv6)

    # Decoder Side
    up1 = concatenate([UpSampling2D(size=(2, 2), data_format="channels_last")(conv6), conv5], axis=3)
    conv7 = Conv2D(512, (3, 3), activation='relu', padding='same', data_format="channels_last")(up1)
    conv7 = BatchNormalization(axis=-1)(conv7)

    up2 = concatenate([UpSampling2D(size=(2, 2), data_format="channels_last")(conv7), conv4], axis=3)
    conv8 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format="channels_last")(up2)
    conv8 = BatchNormalization(axis=-1)(conv8)

    up3 = concatenate([UpSampling2D(size=(2, 2), data_format="channels_last")(conv8), conv3], axis=3)
    conv9 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_last")(up3)
    conv9 = BatchNormalization(axis=-1)(conv9)

    up4 = concatenate([UpSampling2D(size=(2, 2), data_format="channels_last")(conv9), conv2], axis=3)
    conv10 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_last")(up4)
    conv10 = BatchNormalization(axis=-1)(conv10)

    up5 = concatenate([UpSampling2D(size=(2, 2), data_format="channels_last")(conv10), conv1], axis=3)
    conv11 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format="channels_last")(up5)
    conv11 = BatchNormalization(axis=-1)(conv11)

    conv12 = Conv2D(3, (1, 1), activation='sigmoid', data_format="channels_last")(conv11)
    fcn = Model(input=[inputs, model_path], output=[conv12, model_path])

    return fcn

def get_generator(img_shape):
    model = generator(img_shape)
    model.summary()
    return model

def get_discriminator(img_shape, disc_shape, patch_num):
    model = discriminator(img_shape, disc_shape, patch_num)
    model.summary()
    return model

def get_GAN(generator, discriminator, img_shape, patch_size):
    model = generator_discriminator(generator, discriminator, img_shape, patch_size)
    return model