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
import logging

import numpy as np
from PIL.Image import Image

from skimage.transform import resize
from tensorflow.keras.applications.inception_resnet_v2 import (  # pylint: disable=import-error
    InceptionResNetV2,
)
from pycloud.core import PyCloud
from pycloud.config import configure_logging
from PIL import Image

from classess import IMAGENET_LABELS
from helpers import get_image, show_image

configure_logging()

LOGGER = logging.getLogger("ClassificationDemo")

CLOUD = PyCloud.get_instance()


@CLOUD.init_service("classificator")
def create_model():
    """Create image classification object"""
    model = InceptionResNetV2(
            include_top=True,
            weights="imagenet",
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=1000,
        )
    data = {"model": model}
    return data


@CLOUD.endpoint("classificator")
def predict(image):
    """ Run prediction on model"""
    predictions = CLOUD.initialized_data()["model"].predict(image)
    return postprocess(predictions)


@CLOUD.endpoint("classificator")
def pre_process(image):
    """Resize image to size expected by model"""
    if isinstance(image, bytes):
        image = Image.open(io.BytesIO(image))
        image = np.array(image.convert("RGB"))
    LOGGER.info("Input image shape: %s", len(image.shape))
    dim = np.array((299, 299, 3))
    image = resize(image, dim)
    image = np.expand_dims(image, axis=0)
    return predict(image)


@CLOUD.init_service("postprocessing")
def init_postprocessing():
    return {"accuracy_threshold": 0.7 }


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


@CLOUD.endpoint("api")
def classify(image):
    """Put image into pipeline"""
    result = pre_process(image)
    return result


@CLOUD.endpoint("api")
def set_threshold(threshold):
    CLOUD.broadcast(set_acc_threshold, threshold)


def build_app():
    """Test and build graph"""

    init_postprocessing()
    create_model()
    set_threshold(0.6)
    for path in ["images/laptop.jpeg", "images/winter.jpeg"]:
        image, path = get_image(path)
        LOGGER.info("Testing with image: %s", path)
        detection = classify(image)
        print("Test classification result: {}".format(detection))

    CLOUD.configure_service("classificator", environment={"MAX_GRPC_WORKERS": 1 })
    CLOUD.expose_service("api")
    CLOUD.set_basic_auth_credentials("pycloud", "demo")


CLOUD.build(build_app)
