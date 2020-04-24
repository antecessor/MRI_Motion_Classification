import os

import cv2
import keras
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow_core.python.keras.layers import Lambda
from tensorflow_core.python.keras.saving.save import load_model

from Utils.DataUtils.LoadingUtils import readImage


def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))


def target_category_loss_output_shape(input_shape):
    return input_shape


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                   tf.cast(op.inputs[0] > 0., dtype)


def compile_saliency_function(model, activation_layer='block5_conv3'):
    input_img = model.input
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    layer_output = layer_dict[activation_layer].output
    max_output = K.max(layer_output, axis=3)
    saliency = K.gradients(K.sum(max_output), input_img)[0]
    return K.function([input_img, K.learning_phase()], [saliency])


def modify_backprop(model, name):
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': name}):

        # get layers that have an activation
        layer_dict = [layer for layer in model.layers[1:]
                      if hasattr(layer, 'activation')]

        # replace relu activation
        for layer in layer_dict:
            if layer.activation == keras.activations.relu:
                layer.activation = tf.nn.relu

        # re-instanciate a new model
        new_model = load_model('../../motionClassificationModel.h5')
    return new_model


def deprocess_image(x):
    '''
    Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    '''
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    # if K.image_dim_ordering() == 'th':
    #     x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def grad_cam(input_model, image, category_index, layer_name):
    model = input_model

    nb_classes = 5
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    model.add(Lambda(target_layer,
                     output_shape=target_category_loss_output_shape))

    loss = K.sum(model.layers[-1].output, keepdims=True)
    conv_output = [l for l in model.layers if l.name == layer_name][0].output
    grads = normalize(K.gradients(loss, conv_output)[0])
    gradient_function = K.function([model.input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.ones(output.shape[0: 2], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam, (256, 256))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    # # Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)

    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_VIRIDIS)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap


def createPairImages(dataFrame):
    X = []
    Y = []
    for index, row in dataFrame.iterrows():
        if row[0] != 0:
            name = row[1]
            imageWithMotion = row[3]
            number = row[2]
            df1 = dataFrame[(dataFrame.motionNorm == 0) & (dataFrame.name == name) & (dataFrame.number == number)]
            imageWithoutMotion = df1["image"].values[0]
            if imageWithoutMotion.shape != (256, 256) or imageWithMotion.shape != (256, 256):
                continue
            X.append(imageWithMotion)
            Y.append(imageWithoutMotion)
    return X, Y


def getImages():
    baseDir = "E:\Workspaces\PhillipsProject\Data\generated/"
    images = os.listdir(baseDir)
    dataFrame = pd.DataFrame(columns=["motionNorm", "name", "number", "image"])
    motionValue = []
    names = []
    numbers = []
    imageMats = []
    for image in images:
        if str(image).__contains__(".tiff") and str(image).__contains__("T1"):
            imageMat = readImage(baseDir + image, show=False)

            params = image.split("_")
            displacementNorm = float(params[0])
            rotationNorm = float(params[1])
            motionValue.append(np.sqrt(np.power(displacementNorm, 2) + np.power(rotationNorm, 2)))
            names.append(params[2])
            numbers.append(params[3])
            imageMats.append(imageMat)
            print("load image {}".format(image))
    dataFrame['motionNorm'] = motionValue
    dataFrame['name'] = names
    dataFrame['number'] = numbers
    dataFrame['image'] = imageMats

    X, Y = createPairImages(dataFrame)
    return X, Y


X, Y = getImages()

savedName = "Motion4_1"
index = 0
for index in range(120, 181):
    preprocessed_input = X[index]
    # preprocessed_input = cv2.resize(preprocessed_input, (224, 224))
    preprocessed_input = np.reshape(preprocessed_input, (-1, 256, 256, 1))
    model = load_model('../../motionClassificationModel.h5')
    predictions = model.predict(preprocessed_input)

    predicted_class = np.argmax(predictions)
    cam, heatmap = grad_cam(model, preprocessed_input, predicted_class, "conv2d_3")
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(Y[index], cmap='gray')
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(X[index], cmap='gray')
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(cam)

    fig.savefig('Image_{}.png'.format(index))
# cv2.imwrite(savedName + ".jpg", cam)

# register_gradient()
# guided_model = modify_backprop(model, 'GuidedBackProp')
# saliency_fn = compile_saliency_function(guided_model, "activation_3")
# saliency = saliency_fn([preprocessed_input, 0])
# gradcam = saliency[0] * heatmap[..., np.newaxis]
# cv2.imwrite("guided_gradcam2.jpg", deprocess_image(gradcam))
