
import os
import math
from tqdm import tqdm

from PIL import Image
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.utils import shuffle

from models import *
from utils import *


IMAGE_DIRECTORY = 'data/faces/img_align_celeba/'
TOTAL_IMAGES = len(os.listdir(IMAGE_DIRECTORY))
print(TOTAL_IMAGES)

WIDTH, HEIGHT = 256, 256
LATENT_DIM = 100

attribute_data = pd.read_csv('data/list_attr_celeba.csv')
images, attributes = read_data(IMAGE_DIRECTORY, WIDTH, HEIGHT, attribute_data)


# show_images(images)

#Getting the models

generator = get_generator()
discriminator = get_discriminator()
discriminator.trainable = False

#Building the GAN
gan_input = Input(shape=(LATENT_DIM, ))
generator_output = generator(gan_input)
gan_output = discriminator(generator_output)
gan = Model(gan_input, gan_output)

optimizer = Adam(lr=.0001, clipvalue=1.0, decay=1e-8)
gan.compile(optimizer=optimizer, loss='binary_crossentropy')


X, y = shuffle(X, y, random_state=0)

# latent = generate_latent_points(LATENT_DIM, 25, 40)
first_generated = generate_fake_samples(generator, LATENT_DIM, 25)
show_images(first_generated)