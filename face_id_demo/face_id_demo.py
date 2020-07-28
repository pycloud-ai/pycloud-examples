import os
import io
import numpy as np

from itertools import compress
import face_recognition

from pycloud.core import PyCloud
from PIL import Image

EXISTS_IN_DATABASE = "Name {} already exists in database"

NO_FACE_DETECTED = "No face detected"

TOO_MANY_FACES = "Too many faces"

CLOUD = PyCloud.get_instance()

dir_path = os.path.dirname(os.path.realpath(__file__))

@CLOUD.init_service("face_recognition")
def initialize_service():
    known_encodings = []
    known_names = []
    return (known_encodings, known_names)


@CLOUD.endpoint(service_id="encoder")
def encode(image):
    return face_recognition.face_encodings(image)


@CLOUD.endpoint(service_id="face_recognition")
def preprocess(image):
    if isinstance(image, np.ndarray):
        return image
    if isinstance(image, bytes):
        image = Image.open(io.BytesIO(image))
        image = np.array(image.convert("RGB"))
        return image
    raise Exception("Unknown image format")


@CLOUD.endpoint("face_recognition")
def register(image, name):
    image = CLOUD.call(preprocess, image)
    known_encodings, known_names = CLOUD.initialized_data()
    encodings = CLOUD.call(encode, image)
    if len(encodings) == 0:
        return NO_FACE_DETECTED
    if len(encodings) > 1:
        return TOO_MANY_FACES
    # TODO: possible race condition if adding two guys in parallel
    if name in known_names:
        return EXISTS_IN_DATABASE.format(name)
    known_encodings.append(encodings[0])
    known_names.append(name)
    return "OK"


@CLOUD.endpoint("face_recognition")
def recognize(image):
    known_encodings, known_names = CLOUD.initialized_data()
    image = CLOUD.call(preprocess, image)
    unknown_encoding = CLOUD.call(encode, image)
    matches = face_recognition.compare_faces(known_encodings, unknown_encoding[0])
    result = list(compress(known_names, matches))
    return result


if __name__ == "__main__":
    obama_image = face_recognition.load_image_file(os.path.join(dir_path, "obama.jpeg"))
    unknown_image = face_recognition.load_image_file(os.path.join(dir_path, "obama2.jpeg"))
    unknown_image2 = face_recognition.load_image_file(os.path.join(dir_path, "krzych.jpeg"))
    two_guys = face_recognition.load_image_file(os.path.join(dir_path, "two_guys.jpeg"))
    chuck_norris = face_recognition.load_image_file(os.path.join(dir_path, "chuck.jpeg"))
    faceless = face_recognition.load_image_file(os.path.join(dir_path, "faceless.png"))

    initialize_service()
    assert register(obama_image, "Barrack Obama") == "OK"
    assert register(chuck_norris, "Chuck Norris") == "OK"
    assert recognize(unknown_image) == ["Barrack Obama"]
    with open(os.path.join(dir_path, "obama2.jpeg"), "rb") as file:
        image_bytes = file.read()

    assert recognize(image_bytes) == ["Barrack Obama"]
    assert recognize(unknown_image2) == []

    assert register(two_guys, ["Chuck Norris", "Liam Neeson"]) == TOO_MANY_FACES
    assert register(faceless, []) == NO_FACE_DETECTED
    assert "already exists" in register(chuck_norris, "Chuck Norris")

    assert recognize(two_guys) == ["Chuck Norris"]  # one guy only detected

    CLOUD.configure_service("face_recognition", exposed=True, package_deps=["cmake"])
    CLOUD.set_basic_auth_credentials("pycloud", "demo")
    CLOUD.save()
