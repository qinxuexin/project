from keras.backend import clear_session
from keras.models import Model
from keras.regularizers import l2
from keras.layers import Input, Reshape, Dense, Lambda, Activation
from keras.applications import *


clear_session()

input_tensor = Input(shape=[224, 224, 3])


def _outer_product(x):  # Calculate outer-products of two tensors
    return backend.batch_dot(x[0], x[1], axes=[1, 1]) / x[0].get_shape().as_list()[1]


def _signed_sqrt(x):  # Calculate element-wise signed square-root.
    return backend.sign(x) * backend.sqrt(backend.abs(x) + 1e-9)


def _l2_normalize(x, axis=-1):  # Calculate L2 normalization.
    return backend.l2_normalize(x, axis=axis)


def bcnn(
        all_trainable=False,
        no_class=197,
        no_last_layer_backbone=17,
        decay_weight_rate=0.0,
        name_initializer='glorot_normal',
        name_activation='softmax',
):
    resnet_50 = resnet50(
        input_tensor=input_tensor,
        include_top=False,
        weights='imagenet')
    # Pre-trained weights

    # Pre-trained weights
    for layer in resnet_50.layers:
        layer.trainable = all_trainable

    # Extract features form detector
    model_detector = resnet_50
    output_detector = model_detector.layers[no_last_layer_backbone].output
    shape_detector = model_detector.layers[no_last_layer_backbone].output_shape

    # Extract features from extractor
    model_extractor = resnet_50
    output_extractor = model_extractor.layers[no_last_layer_backbone].output
    shape_extractor = model_extractor.layers[no_last_layer_backbone].output_shape
    # print(shape_detector[1])
    # Reshape tensor to (minibatch_size, total_pixels, filter_size)
    output_detector = Reshape(
        [shape_detector[1] * shape_detector[2], shape_detector[-1]])(output_detector)
    output_extractor = Reshape(
        [shape_extractor[1] * shape_extractor[2], shape_extractor[-1]])(output_extractor)

    x = Lambda(_outer_product)([output_detector, output_extractor])
    x = Reshape([shape_detector[-1] * shape_extractor[-1]])(x)
    x = Lambda(_signed_sqrt)(x)
    x = Lambda(_l2_normalize)(x)

    x = Dense(
        units=no_class,
        kernel_initializer=name_initializer,
        kernel_regularizer=l2(decay_weight_rate))(x)
    output_tensor = Activation(name_activation)(x)

    model_bcnn = Model(inputs=[input_tensor], outputs=[output_tensor])
    return model_bcnn


if __name__ == "__main__":
    pass
