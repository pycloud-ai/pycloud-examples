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
import logging

import numpy as np

from skimage.transform import resize
from tensorflow.keras.applications.inception_resnet_v2 import (  # pylint: disable=import-error
    InceptionResNetV2,
)
from pycloud.core import PyCloud
from pycloud.helpers import is_this_executed_on_runner
from pycloud.config import configure_logging

from classess import IMAGENET_LABELS

if not is_this_executed_on_runner():
    from helpers import (  # pylint: disable=no-name-in-module
        show_image,
        get_random_image,
    )
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
def classify(image):
    """ Run prediction on model"""
    predictions = CLOUD.initialized_data()["model"].predict(image)
    return CLOUD.call(postprocess, predictions)


@CLOUD.endpoint("classificator")
def pre_process(image):
    """Resize image to size expected by model"""
    LOGGER.info("Input image shape: %s", len(image.shape))
    dim = np.array((299, 299, 3))
    image = resize(image, dim)
    image = np.expand_dims(image, axis=0)
    return CLOUD.call(classify, image)


@CLOUD.endpoint("asynchronous-consumer", protocols=["AMQP"])
def asynchronous_consumer(message):
    LOGGER.info("Just received asynchronous message :%s", message)


@CLOUD.endpoint("postprocessing")
def postprocess(results):
    """ Convert model output to string label"""
    index = np.argmax(results)
    LOGGER.info("Detected class number: %d", index)
    label = IMAGENET_LABELS[index]
    CLOUD.message(asynchronous_consumer, label)
    return label


@CLOUD.endpoint("api", protocols=["GRPC"])
def api(image):
    """Run whole pipeline"""
    result = CLOUD.call(pre_process, image)
    return result


def test_and_deploy():
    """Exec demo"""
    no_images = 1
    images = []
    for _ in range(no_images):
        img, keywords = get_random_image()
        images.append((img, keywords))

    create_model()
    for image, description in images:
        LOGGER.info("Testing with image: %s", description)
        detection = api(image)
        show_image(image, label=detection)
        print("Test classification result: {}".format(detection))

    CLOUD.expose_service("api")
    CLOUD.set_basic_auth_credentials("pycloud", "demo")
    CLOUD.save()


if __name__ == "__main__":
    test_and_deploy()
