#!/usr/bin/env python

import numpy as np

from model import TFLiteModel
import argparse

p = argparse.ArgumentParser()
p.add_argument('model', help='path to model')
p.add_argument('image', help='path to input')
p.add_argument('outfile', help='where to store output')
args = p.parse_args()

## We use keras for loading the image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


def load_image_data(image_path, input_shape):
    image = load_img(image_path, target_size=input_shape[1:-1])
    np_image = img_to_array(image)
    image_batch = np.expand_dims(np_image, axis=0)
    return image_batch


# figure out shape of the input
model = TFLiteModel(args.model)
input_shape = model.get_input().shape
data = load_image_data(args.image, input_shape)


# write input image to a file
with open(args.outfile, 'w') as f:
    for x in data.flatten():
        f.write(int(x))
        f.write('\n')
