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

import os
import logging
import cv2
from PIL import Image

LOGGER = logging.getLogger("DemoHelpers")

directory = os.path.dirname(os.path.realpath(__file__))


def get_image(path):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    img = cv2.imread(os.path.join(dir_path, path))
    img = resize_image(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, path


def resize_image(img):
    while img.nbytes > 1000000:
        img = cv2.resize(img, (int(img.shape[1] / 1.1), int(img.shape[0] // 1.1)))
    return img


def show_image(img, label):
    """Display image on the screen"""
    image = label_image(img, label)
    pimg = Image.fromarray(image)
    pimg.show("")
    # cv2.imshow("", image)
    # cv2.waitKey(250)


def label_image(img, text):
    """Print label on image"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottom_left_corner_of_text = (10, 50)
    font_scale = 1
    font_color = (10, 10, 10)
    line_type = 2
    cv2.putText(
        img, text, bottom_left_corner_of_text, font, font_scale, font_color, line_type
    )
    return img
