import sys
import PIL.Image
import numpy as np

from pycloud_client.client import GRPCClient

def load_image(file):
    image = PIL.Image.open(file)
    image = image.convert("RGB")
    return np.array(image)

grpc_host = sys.argv[1]
grpc_port = sys.argv[2]
image_file = sys.argv[3]
name = sys.argv[4]

client = GRPCClient(grpc_host, grpc_port)

response = client.request("add_known_encoding@face_recognition_demo", load_image(image_file), name)
print(response)
