import tensorflow as tf
from keras import Input
from keras.layers import Dense, Reshape, LeakyReLU, Conv2D, Conv2DTranspose, Flatten, Dropout, Concatenate
from keras.models import Model
from keras.optimizers import Adam

def get_generator(latent_dim = 100, channels = 3, n_attributes=40):
    gen_input = Input(shape=(latent_dim, ))
    #100,1
    
    label_input = Input(shape=(n_attributes,))
    label_intermediate = Dense(48)(label_input)
    
    merge = Concatenate()([gen_input, label_intermediate])

    x = Dense(128 * 16 * 16)(merge)
    x = LeakyReLU()(x)
    x = Reshape((16, 16, 128))(x)
    
    x = Conv2D(256, kernel_size = (5, 5), padding='same')(x)
    x = LeakyReLU()(x)
    #16 * 16 * 256

    x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = LeakyReLU()(x)
    #32*32*256

    x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = LeakyReLU()(x)
    #64*64*256

    x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = LeakyReLU()(x)
    #128*128*256

    x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = LeakyReLU()(x)
    #256*256*256

    x = Conv2D(512, kernel_size=(5, 5), padding='same')(x)
    x = LeakyReLU()(x)
    #256*256*512

    x = Conv2D(512,kernel_size=(5, 5), padding='same')(x)
    x = LeakyReLU()(x)
    #256*256*512

    output = Conv2D(channels, kernel_size=(7, 7), activation='tanh', padding='same')(x)
    #256*256*3

    generator = Model([gen_input, label_input], output)

    return generator

def get_discriminator(height, width, channels, n_attributes=40):
    disc_input = Input(shape=(height, width, channels))

    label_input = Input(shape=(n_attributes,))
    
    n_nodes = height * width
    label_intermediate = Dense(n_nodes, activation='relu')(label_input)
    merge = Concatenate()([disc_input, label_intermediate])
    
    x = Conv2D(256, 3)(disc_input)
    x = LeakyReLU()(x)
    
    x = Conv2D(256, 4, strides=2)(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(256, 4, strides=2)(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(256, 4, strides=2)(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(256, 4, strides=2)(x)
    x = LeakyReLU()(x)
    
    x = Flatten()(x)
    x = Dropout(0.4)(x)
    
    x = Dense(1, activation='sigmoid')(x)
    discriminator = Model([disc_input, label_input], x)
    
    optimizer = Adam(
        lr=.0001,
        decay=1e-8
    )
    
    discriminator.compile(
        optimizer=optimizer,
        loss='binary_crossentropy'
    )
    
    return discriminator