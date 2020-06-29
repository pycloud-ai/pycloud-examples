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
import random
import sys

from pycloud_client.client import GRPCClient, HTTPClient

HOST = sys.argv[1]

PORT_GRPC = sys.argv[2]

PORT_HTTP = sys.argv[3]

client_grpc = GRPCClient(HOST, PORT_GRPC, client_id="client_grpc")
client_http = HTTPClient(HOST, PORT_HTTP, client_id="client_http")

for _ in range(100000):
    a = random.randrange(1, 10)
    b = random.randrange(1, 10)
    op = random.choice(["add", "multiply"])
    client = random.choice([client_http, client_grpc])
    result = client.request("compute@demo", op, a, b)
    print("{} {} {} = {}".format(a, op, b, result))
