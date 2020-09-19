import math

import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm


def read_data(image_dir, width, height, attribute_data):
    images = []
    attributes = []
    for i in tqdm(range(1000)):
        image_id, current = attribute_data.iloc[i, 0], attribute_data.iloc[i, 1:]
        current[current < 0] = 0
        image = cv2.imread(image_dir+image_id)
        image = image[:, :, [2, 1, 0]]
        image = cv2.resize(image, (width, height), interpolation = cv2.INTER_CUBIC)
        images.append(image)
        attributes.append(current)
    images = np.array(images)
    return images, attributes

def show_images(images):
    size = min(int(math.sqrt(len(images))), 5)
    plt.figure(1, figsize=(10, 10))
    for i in range(25):
        plt.subplot(size, size, i+1)
        plt.imshow(images[i])
        plt.axis('off')
    plt.show()


def generate_real_samples(dataset, n_samples):
	# split into images and labels
	images, labels = dataset
	# choose random instances
	ix = np.randint(0, images.shape[0], n_samples)
	# select images and labels
	X, labels = images[ix], labels[ix]
	# generate class labels
	y = np.ones((n_samples, 1))
	return [X, labels], y


def generate_latent_points(latent_dim, n_samples, n_attributes):
    # generate points in the latent space
    x_input = np.randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    # generate labels
    labels = []
    for _ in range(n_samples):
        labels.append(np.randint(0, 2, n_attributes))
    return [z_input, labels]
 
# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
	# generate points in latent space
	z_input, labels_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	images = generator.predict([z_input, labels_input])
	# create class labels
	y = np.zeros((n_samples, 1))
	return [images, labels_input], y
