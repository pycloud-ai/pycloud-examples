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
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

CLOUD = PyCloud.get_instance()


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
    unknown_encodings = CLOUD.call(encode, image)
    all_matches = None
    for encoding in unknown_encodings:
        matches = face_recognition.compare_faces(known_encodings, encoding)
        if all_matches is None:
            all_matches = matches
        else:
            all_matches = [x | y for (x, y) in zip(all_matches, matches)]
    result = list(compress(known_names, all_matches))
    return result


if __name__ == "__main__":
    OBAMA_IMAGE = face_recognition.load_image_file(os.path.join(DIR_PATH, "obama.jpeg"))
    UNKNOWN_IMAGE = face_recognition.load_image_file(os.path.join(DIR_PATH, "obama2.jpeg"))
    UNKNOWN_IMAGE2 = face_recognition.load_image_file(os.path.join(DIR_PATH, "krzych.jpeg"))
    TWO_GUYS = face_recognition.load_image_file(os.path.join(DIR_PATH, "two_guys.jpeg"))
    CHUCK_NORRIS = face_recognition.load_image_file(os.path.join(DIR_PATH, "chuck.jpeg"))
    LIAM_NEESON = face_recognition.load_image_file(os.path.join(DIR_PATH, "liam.jpeg"))
    FACELESS = face_recognition.load_image_file(os.path.join(DIR_PATH, "faceless.png"))

    CLOUD.start_local_run()
    initialize_service()

    assert register(OBAMA_IMAGE, "Barrack Obama") == "OK"
    assert register(CHUCK_NORRIS, "Chuck Norris") == "OK"
    assert recognize(UNKNOWN_IMAGE) == ["Barrack Obama"]
    with open(os.path.join(DIR_PATH, "obama2.jpeg"), "rb") as file:
        IMAGE_BYTES = file.read()
    assert recognize(IMAGE_BYTES) == ["Barrack Obama"]
    assert recognize(UNKNOWN_IMAGE2) == []

    assert register(TWO_GUYS, ["Chuck Norris", "Liam Neeson"]) == TOO_MANY_FACES
    assert register(FACELESS, []) == NO_FACE_DETECTED
    assert "already exists" in register(CHUCK_NORRIS, "Chuck Norris")

    assert recognize(TWO_GUYS) == ["Chuck Norris"]  # one guy only detected
    assert register(LIAM_NEESON, "Liam Neeson") == "OK"
    assert recognize(TWO_GUYS) == ["Chuck Norris", "Liam Neeson"]

    CLOUD.configure_service("face_recognition", exposed=True, package_deps=["cmake"])
    CLOUD.set_basic_auth_credentials("pycloud", "demo")
    CLOUD.end_local_run()
