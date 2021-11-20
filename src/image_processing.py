import cv2
import numpy as np
from PIL import Image

import tensorflow as tf
import tensorflow.keras.layers as L
import efficientnet.tfkeras as efn


def grayscale():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    r = img_arr[:, :, 0]
    g = img_arr[:, :, 1]
    b = img_arr[:, :, 2]
    new_arr = r.astype(int) + g.astype(int) + b.astype(int)
    new_arr = (new_arr/3).astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


class ProcessingStep:
    def __init__(self):
        pass

    def apply(self, data):
        pass


class PrepareStep(ProcessingStep):
    def __init__(self):
        pass

    def apply(self, data):
        img = data
        img = cv2.resize(img/255.0, (512, 512)).reshape(-1, 512, 512, 3).astype(np.float32)
        return img


class HealthModel(ProcessingStep):
    def __init__(self):
        self.model = tf.keras.Sequential([efn.EfficientNetB7(input_shape=(512, 512, 3),
                                                             weights=None,
                                                             include_top=False),
                                          L.GlobalAveragePooling2D(),
                                          L.Dense(4, activation='softmax')])
        self.model.load_weights("../data/models/health/en_v1/chekpoint.h5")

    def apply(self, data):
        img = data
        preds = self.model.layers[2]( self.model.layers[1]( self.model.layers[0](img)))
        return preds.numpy()[0]

class ImageProcessing(ProcessingStep):
    def __init__(self):
        self.steps = [
            PrepareStep(),
            HealthModel(),
        ]

    def apply(self, data):
        for s in self.steps:
            data = s.apply(data)

        return data