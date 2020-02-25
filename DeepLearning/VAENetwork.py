import os

from keras import backend as K
from keras.backend import binary_crossentropy
from keras.layers import Conv2D, Flatten, Lambda
from keras.layers import Dense, Input
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras.utils import plot_model


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# then z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling
        fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def trainOrGetTrained(x_train, y_train, x_test, y_test, mse, modelWeightsName):
    image_size = x_train.shape[1]
    input_shape = (image_size, image_size, 1)
    batch_size = 32
    kernel_size = 3
    filters = 16
    latent_dim = 16
    epochs = 30

    # VAE model = encoder + decoder
    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    target = Input(shape=input_shape, name='encoder_input')
    x = inputs
    for i in range(2):
        filters *= 2
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   activation='relu',
                   strides=2,
                   padding='same')(x)

    # shape info needed to build decoder model
    shape = K.int_shape(x)

    # generate latent vector Q(z|X)
    x = Flatten()(x)
    x = Dense(16, activation='relu')(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary
    # with the TensorFlow backend
    z = Lambda(sampling,
               output_shape=(latent_dim,),
               name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()
    plot_model(encoder,
               to_file='vae_cnn_encoder.png',
               show_shapes=True)

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(shape[1] * shape[2] * shape[3],
              activation='relu')(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)

    for i in range(2):
        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            activation='relu',
                            strides=2,
                            padding='same')(x)
        filters //= 2

    outputs = Conv2DTranspose(filters=1,
                              kernel_size=kernel_size,
                              activation='sigmoid',
                              padding='same',
                              name='decoder_output')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    plot_model(decoder,
               to_file='vae_cnn_decoder.png',
               show_shapes=True)

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae')
    # VAE loss = mse_loss or xent_loss + kl_loss
    if mse:
        reconstruction_loss = mse(K.flatten(target), K.flatten(outputs))
    else:
        reconstruction_loss = binary_crossentropy(K.flatten(target),
                                                  K.flatten(outputs))

    reconstruction_loss *= image_size * image_size
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='rmsprop')
    vae.build(input_shape)
    vae.summary()
    plot_model(vae, to_file='vae_cnn.png', show_shapes=True)

    if os.path.exists(modelWeightsName):
        vae = vae.load_weights(modelWeightsName)
    else:
        # train the autoencoder

        vae.fit(x=x_train,
                epochs=epochs,
                batch_size=batch_size)
        vae.save_weights(modelWeightsName)

# def predictDecoder( latentSpace):
#     x_decoded = decoder.predict(latentSpace)
#     return x_decoded[0].reshape(image_size, image_size)
#
# def predictEncoder( images):
#     res = encoder.predict(images, batch_size=batch_size)
#     return res[2]
