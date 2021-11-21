import cv2
import numpy as np

import tensorflow as tf
import tensorflow.keras.layers as L
import efficientnet.tfkeras as efn

from mrcnn.config import Config as mcConfig
from mrcnn import model as mcmodel


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

class CustomConfig(mcConfig):
    NAME = "strawberry"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.9

class InferenceConfig(CustomConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class SegmentationStep(ProcessingStep):
    def __init__(self):
        self.model = mcmodel.MaskRCNN(mode="inference",
                                      config=CustomConfig(),
                                      model_dir="../data/models/mskrcnn/v1/")
        self.model.load_weights("../data/models/mskrcnn/v1/chekpoint009.h5", by_name=True)
        # self.model.keras_model._make_predict_function()
        # self.graph = tf.compat.v1.get_default_graph()
        pass

    def apply(self, data):
        img = data['img']
        preds = self.model.detect([img], verbose=1)
        data['seg_preds'] = preds
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
            SegmentationStep(),
            HealthModel(),
            PhaseModel(),
        ]

    def apply(self, data):
        for s in self.steps:
            data = s.apply(data)

        return data