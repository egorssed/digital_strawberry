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


class PreparationStep(ProcessingStep):
    def __init__(self):
        pass

    def apply(self, data):
        img = data['img']
        # img = cv2.resize(img/255.0, (512, 512)).reshape(-1, 512, 512, 3).astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data['img'] = img
        return data


class PhaseModel(ProcessingStep):
    def __init__(self):
        self.model = tf.keras.Sequential([efn.EfficientNetB7(input_shape=(256, 256, 3),
                                                             weights=None,
                                                             include_top=False),
                                          L.GlobalAveragePooling2D(),
                                          L.Dense(3, activation='softmax')])
        self.model.load_weights("../data/models/phase/en_v1/chekpoint.h5")

    def apply(self, data):
        img = data['img']
        img = cv2.resize(img/255.0, (256, 256)).reshape(-1, 256, 256, 3).astype(np.float32)
        preds = self.model.layers[2]( self.model.layers[1]( self.model.layers[0](img)))
        data['phase_preds'] = preds.numpy()[0]
        return data


class HealthModel(ProcessingStep):
    def __init__(self):
        self.model = tf.keras.Sequential([efn.EfficientNetB7(input_shape=(512, 512, 3),
                                                             weights=None,
                                                             include_top=False),
                                          L.GlobalAveragePooling2D(),
                                          L.Dense(4, activation='softmax')])
        self.model.load_weights("../data/models/health/en_v1/chekpoint.h5")

    def apply(self, data):
        img = data['img']
        img = cv2.resize(img/255.0, (512, 512)).reshape(-1, 512, 512, 3).astype(np.float32)
        preds = self.model.layers[2]( self.model.layers[1]( self.model.layers[0](img)))
        data['health_preds'] = preds.numpy()[0]
        return data


class ImageProcessing(ProcessingStep):
    def __init__(self):
        self.steps = [
            PreparationStep(),
            HealthModel(),
            PhaseModel(),
        ]

    def apply(self, data):
        for s in self.steps:
            data = s.apply(data)

        return data