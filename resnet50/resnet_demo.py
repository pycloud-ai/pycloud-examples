#
# Copyright (C) 2020 PyCloud - All Rights Reserved
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Classify images using pretrained keras inception resnet"""
import io
import os
import cv2
import time
import logging

import numpy as np
from PIL.Image import Image

from cv2 import resize
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from tensorflow.keras.models import save_model, load_model
from pycloud.core import PyCloud
from pycloud.config import configure_logging
from PIL import Image

from classess import IMAGENET_LABELS
from helpers import get_image, show_image

configure_logging()

LOGGER = logging.getLogger("ClassificationDemo")

CLOUD = PyCloud.get_instance()


@CLOUD.endpoint("api")
def classify(image):
    """Put image into pipeline"""
    result = preprocess(image)
    return result


@CLOUD.endpoint("api")
def set_threshold(threshold):
    CLOUD.broadcast(set_acc_threshold, threshold)


@CLOUD.init_service("resnet")
def create_model():
    """Create image classification object"""
    file_name = "./resnet50.h5"

    dir_path = os.path.dirname(os.path.realpath(__file__))
    full_path = os.path.join(dir_path, file_name)
    if os.path.exists(full_path):
        model = load_model(
            full_path, custom_objects=None, compile=True
        )
    else:
        model = InceptionResNetV2(
            include_top=True,
            weights="imagenet",
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=1000,
        )
        save_model(
            model, full_path, overwrite=True, include_optimizer=True, save_format=None,
            signatures=None, options=None
        )
    
    data = {"model": model}
    return data


@CLOUD.endpoint("resnet")
def predict(image):
    """ Run prediction on model"""
    t0 = time.time()
    predictions = CLOUD.initialized_data()["model"].predict(image)
    t1 = time.time()
    CLOUD.collect_metric("inference time", ["MAX", "MEAN"], t1-t0)
    return postprocess(predictions)


def resize_image(image):
    dim = (299, 299)
    return resize(image, dim)
   

@CLOUD.endpoint("resnet")
def preprocess(image):
    """Resize image to size expected by model"""
    t0 = time.time()
    if isinstance(image, bytes):
        jpg_as_np = np.frombuffer(image, dtype=np.uint8)
        image = cv2.imdecode(jpg_as_np, flags=1)
    t1 = time.time()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = np.expand_dims(resize_image(image), axis=0)
    t2 = time.time()
    image=preprocess_input(image)
    t3 = time.time()
    result = predict(image)
    t4 = time.time()
    CLOUD.collect_metric("t1", ["MIN","MAX","MEAN"], t1-t0)
    CLOUD.collect_metric("t2", ["MIN","MAX","MEAN"], t2-t1)
    CLOUD.collect_metric("t3", ["MIN","MAX","MEAN"], t3-t2)
    CLOUD.collect_metric("t4", ["MIN","MAX","MEAN"], t4-t3)
    return result


@CLOUD.init_service("postprocessing")
def init_postprocessing():
    return {"accuracy_threshold": 0.5}


@CLOUD.queue_consumer("postprocessing")
def set_acc_threshold(threshold):
    threshold = float(threshold)
    CLOUD.initialized_data()['accuracy_threshold'] = threshold
    CLOUD.collect_metric('Accuracy threshold', ['LAST'], threshold)


@CLOUD.endpoint("postprocessing")
def postprocess(predictions):
    """ Convert model output to string label"""
    index = np.argmax(predictions)
    accuracy = predictions[0][index]
    CLOUD.collect_metric("Accuracy", ["MIN", "MAX"], accuracy)
    LOGGER.info("Detected class number: %d, accuracy %f", index, accuracy)
    if accuracy >= CLOUD.initialized_data()['accuracy_threshold']:
        label = IMAGENET_LABELS[index]
        CLOUD.collect_metric("Classified", ["COUNT"], 1)
    else:
        CLOUD.collect_metric("Not classified", ["COUNT"], 1)
        label = ""
    return label


def build_app():
    """Test and build graph"""

    init_postprocessing()
    create_model()
    set_threshold(0.5)
    for path in ["images/laptop.jpeg", "images/winter.jpeg"]:
        image, path = get_image(path)
        LOGGER.info("Testing with image: %s", path)
        detection = classify(image)
        print("Test classification result: {}".format(detection))

    CLOUD.configure_service("resnet", environment={"MAX_GRPC_WORKERS": 1})
    CLOUD.expose_service("api")
    CLOUD.set_basic_auth_credentials("pycloud", "demo")


CLOUD.build(build_app)
