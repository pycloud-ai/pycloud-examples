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
import sys
import time
import threading

from pycloud_client.client import GRPCClient

from helpers import get_random_image

logging.basicConfig(level=logging.INFO)

LOGGER = logging.getLogger("ClassificationDemo")

HOST = sys.argv[1]
PORT = sys.argv[2]

client = GRPCClient(HOST, PORT, client_id="demo")

NUM_THREADS = 1
if len(sys.argv) > 3:
    NUM_THREADS = int(sys.argv[3])

print("Using {} threads".format(NUM_THREADS))


def send_requests():
    no_images = 5
    images = []
    for _ in range(no_images):
        img, keywords = get_random_image()
        images.append((img, keywords))
    LOGGER.info("Start inference")
    threads = []
    for number in range(NUM_THREADS):
        thread = threading.Thread(target=request_thread, args=(images, number))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

expected_classess = { "laptop.jpeg": "laptop, laptop computer",
                      "winter.jpeg": "snowplow",
                      "knife.jpeg": "can opener",
                      "shoe.jpeg": "running shoe",
                      "orange.jpeg": "orange"
                      }

def request_thread(images, number):
    cntr = 0
    errors = 0
    idx = 0
    while True:
        img, path = images[idx]
        cntr += 1
        idx += 1
        if idx == len(images):
            idx = 0
        t0 = time.time()
        detection = None
        try:
            detection = client.request("api@classification_demo", img)
            t1 = time.time()
            LOGGER.info("Detected class from image :%s: %s, time: %s seconds", path, detection, t1 - t0)
            img_file = path.split("/")[-1]
            if expected_classess[img_file] not in detection:
                errors += 1 
            LOGGER.info("Error ratio: %s", str(100 * errors/cntr)) 
        except Exception as e:
            print("exception during call: {}".format(e))
        

if __name__ == "__main__":
    send_requests()
