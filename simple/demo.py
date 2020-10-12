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

import logging
import time

from pycloud.core import PyCloud

CLOUD = PyCloud.get_instance()

LOGGER = logging.getLogger("SimpleDemo")


@CLOUD.endpoint("compute-service")
def compute(op, a, b):
    if op == "add":
        return add(a, b)
    elif op == "multiply":
        return multiply(a, b)
    else:
        return "unknown op"


@CLOUD.endpoint("add-service", protocols=['HTTP'])
def add(a, b):
    return a + b


@CLOUD.endpoint("multiply-service", protocols=['GRPC'])
def multiply(a, b):
    result = a * b
    CLOUD.enqueue(publish, result)
    return result


@CLOUD.init_service("publish-service")
def initialize_publisher():
    return {"data": "something"}


@CLOUD.queue_consumer("publish-service")
def publish(result):
    data = CLOUD.initialized_data()
    time.sleep(2)
    LOGGER.info("Result is : {}, and the data is: {}".format(result, data))


def build_simple_app():

    initialize_publisher()
    val1 = compute("add", 2, 3)
    assert val1 == 5
    val2 = compute("multiply", 2, 3)
    assert val2 == 6

    CLOUD.expose_service("compute-service")
    CLOUD.set_basic_auth_credentials("pycloud", "demo")


CLOUD.build(build_simple_app)
