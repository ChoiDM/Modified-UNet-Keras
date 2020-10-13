from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.activations import sigmoid


def modified_2d_unet(n_classes=1, input_size=(256, 256, 1), dropout=0.6, base_n_filter=32):
    '''
    Original Pytorch Modified 3D Unet Code : https://github.com/pykao/Modified-3D-UNet-Pytorch/blob/master/model.py
    '''
    inputs = Input(input_size)

    # Level 1 context 
    out = Conv2D(base_n_filter, 3, activation=None, padding='same', kernel_initializer='he_normal', name='conv2d_c1_1')(inputs)
    residual_1 = out
    out = LeakyReLU()(out)
    out = Conv2D(base_n_filter, 3, activation=None, padding='same', kernel_initializer='he_normal', name='conv2d_c1_2')(out)
    out = Dropout(dropout)(out)
    out = LeakyReLU()(out)
    out = Conv2D(base_n_filter, 3, activation=None, padding='same', kernel_initializer='he_normal', name='conv2d_c1_3')(out)

    out += residual_1
    context_1 = LeakyReLU()(out)
    out = InstanceNormalization()(out)
    out = LeakyReLU()(out)
    
    # Level 2 context 
    out = Conv2D(base_n_filter*2, 3, activation=None, strides=(2,2), padding='same', kernel_initializer='he_normal', name='conv2d_c2_1')(out)
    residual_2 = out
    out = InstanceNormalization()(out)
    out = LeakyReLU()(out)
    out = Conv2D(base_n_filter*2, 3, activation=None, padding='same', kernel_initializer='he_normal', name='conv2d_c2_2')(out)
    out = Dropout(dropout)(out)
    out = InstanceNormalization()(out)
    out = LeakyReLU()(out)
    out = Conv2D(base_n_filter*2, 3, activation=None, padding='same', kernel_initializer='he_normal', name='conv2d_c2_3')(out)
    
    out += residual_2
    out = InstanceNormalization()(out)
    out = LeakyReLU()(out)
    context_2 = out

    # Level 3 context 
    out = Conv2D(base_n_filter*4, 3, activation=None, strides=(2,2), padding='same', kernel_initializer='he_normal', name='conv2d_c3_1')(out)
    residual_3 = out
    out = InstanceNormalization()(out)
    out = LeakyReLU()(out)
    out = Conv2D(base_n_filter*4, 3, activation=None, padding='same', kernel_initializer='he_normal', name='conv2d_c3_2')(out)
    out = Dropout(dropout)(out)
    out = InstanceNormalization()(out)
    out = LeakyReLU()(out)
    out = Conv2D(base_n_filter*4, 3, activation=None, padding='same', kernel_initializer='he_normal', name='conv2d_c3_3')(out)
    
    out += residual_3
    out = InstanceNormalization()(out)
    out = LeakyReLU()(out)
    context_3 = out

    # Level 4 context 
    out = Conv2D(base_n_filter*8, 3, activation=None, strides=(2,2), padding='same', kernel_initializer='he_normal', name='conv2d_c4_1')(out)
    residual_4 = out
    out = InstanceNormalization()(out)
    out = LeakyReLU()(out)
    out = Conv2D(base_n_filter*8, 3, activation=None, padding='same', kernel_initializer='he_normal', name='conv2d_c4_2')(out)
    out = Dropout(dropout)(out)
    out = InstanceNormalization()(out)
    out = LeakyReLU()(out)
    out = Conv2D(base_n_filter*8, 3, activation=None, padding='same', kernel_initializer='he_normal', name='conv2d_c4_3')(out)
    
    out += residual_4
    out = InstanceNormalization()(out)
    out = LeakyReLU()(out)
    context_4 = out

    # Level 5 context 
    out = Conv2D(base_n_filter*16, 3, activation=None, strides=(2,2), padding='same', kernel_initializer='he_normal', name='conv2d_c5_1')(out)
    residual_5 = out
    out = InstanceNormalization()(out)
    out = LeakyReLU()(out)
    out = Conv2D(base_n_filter*16, 3, activation=None, padding='same', kernel_initializer='he_normal', name='conv2d_c5_2')(out)
    out = Dropout(dropout)(out)
    out = InstanceNormalization()(out)
    out = LeakyReLU()(out)
    out = Conv2D(base_n_filter*16, 3, activation=None, padding='same', kernel_initializer='he_normal', name='conv2d_c5_3')(out)
    
    out += residual_5
    out = InstanceNormalization()(out)
    out = LeakyReLU()(out)
    out = UpSampling2D(size=(2, 2))(out)
    out = Conv2D(base_n_filter*8, 3, activation=None, padding='same', kernel_initializer='he_normal', name='conv2d_l0_0')(out)
    out = InstanceNormalization()(out)
    out = LeakyReLU()(out)

    out = Conv2D(base_n_filter*8, 3, activation=None, padding='same', kernel_initializer='he_normal', name='conv2d_l0_1')(out)
    out = InstanceNormalization()(out)
    out = LeakyReLU()(out)

    # Level 1 localization
    out = Concatenate(axis=3)([out, context_4])
    out = Conv2D(base_n_filter*16, 3, activation=None, padding='same', kernel_initializer='he_normal', name='conv2d_l1_0')(out)
    out = InstanceNormalization()(out)
    out = LeakyReLU()(out)
    out = Conv2D(base_n_filter*8, 3, activation=None, padding='same', kernel_initializer='he_normal', name='conv2d_l1_1')(out)
    out = InstanceNormalization()(out)
    out = LeakyReLU()(out)
    out = UpSampling2D(size=(2, 2))(out)
    out = Conv2D(base_n_filter*4, 3, activation=None, padding='same', kernel_initializer='he_normal', name='conv2d_l1_2')(out)
    out = InstanceNormalization()(out)
    out = LeakyReLU()(out)

    # Level 2 localization
    out = Concatenate(axis=3)([out, context_3])
    out = Conv2D(base_n_filter*8, 3, activation=None, padding='same', kernel_initializer='he_normal', name='conv2d_l2_0')(out)
    out = InstanceNormalization()(out)
    out = LeakyReLU()(out)
    ds2 = out
    out = Conv2D(base_n_filter*4, 3, activation=None, padding='same', kernel_initializer='he_normal', name='conv2d_l2_1')(out)
    out = InstanceNormalization()(out)
    out = LeakyReLU()(out)
    out = UpSampling2D(size=(2, 2))(out)
    out = Conv2D(base_n_filter*2, 3, activation=None, padding='same', kernel_initializer='he_normal', name='conv2d_l2_2')(out)
    out = InstanceNormalization()(out)
    out = LeakyReLU()(out)

    # Level 3 localization
    out = Concatenate(axis=3)([out, context_2])
    out = Conv2D(base_n_filter*4, 3, activation=None, padding='same', kernel_initializer='he_normal', name='conv2d_l3_0')(out)
    out = InstanceNormalization()(out)
    out = LeakyReLU()(out)
    ds3 = out
    out = Conv2D(base_n_filter*2, 3, activation=None, padding='same', kernel_initializer='he_normal', name='conv2d_l3_1')(out)
    out = InstanceNormalization()(out)
    out = LeakyReLU()(out)
    out = UpSampling2D(size=(2, 2))(out)
    out = Conv2D(base_n_filter, 3, activation=None, padding='same', kernel_initializer='he_normal', name='conv2d_l3_2')(out)
    out = InstanceNormalization()(out)
    out = LeakyReLU()(out)

    # Level 4 localization
    out = Concatenate(axis=3)([out, context_1])
    out = Conv2D(base_n_filter*2, 3, activation=None, padding='same', kernel_initializer='he_normal', name='conv2d_l4_0')(out)
    out = InstanceNormalization()(out)
    out = LeakyReLU()(out)
    out_pred = Conv2D(n_classes, 3, activation=None, padding='same', kernel_initializer='he_normal', name='conv2d_l4_1')(out)

    ds2_1x1_conv = Conv2D(n_classes, 1, activation=None, padding='same', kernel_initializer='he_normal', name='ds2_1x1_conv')(ds2)
    ds1_ds2_sum_upscale = UpSampling2D(size=(2, 2))(ds2_1x1_conv)
    ds3_1x1_conv = Conv2D(n_classes, 1, activation=None, padding='same', kernel_initializer='he_normal', name='ds3_1x1_conv')(ds3)
    ds1_ds2_sum_upscale_ds3_sum = ds1_ds2_sum_upscale + ds3_1x1_conv
    ds1_ds2_sum_upscale_ds3_sum_upscale = UpSampling2D(size=(2, 2))(ds1_ds2_sum_upscale_ds3_sum)

    out = out_pred + ds1_ds2_sum_upscale_ds3_sum_upscale
    out = sigmoid(out)

    model = Model(inputs=inputs, outputs=out)

    return model
