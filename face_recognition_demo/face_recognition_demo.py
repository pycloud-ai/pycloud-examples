from itertools import compress
import face_recognition

from pycloud.core import PyCloud

CLOUD = PyCloud.get_instance()


@CLOUD.init_service("face_recognition")
def initialize_service():
    known_encodings = []
    known_names = []
    return (known_encodings, known_names)


@CLOUD.endpoint(service_id="encoder")
def encode(image):
    return face_recognition.face_encodings(image)[0]


@CLOUD.endpoint("face_recognition")
def add_known_encoding(image, name):
    known_encodings, known_names = CLOUD.initialized_data()
    # TODO: possible race condition
    known_encodings.append(CLOUD.call(encode, image))
    known_names.append(name)


@CLOUD.endpoint("face_recognition")
def recognize(image):
    known_encodings, known_names = CLOUD.initialized_data()
    unknown_encoding = CLOUD.call(encode, image)
    matches = face_recognition.compare_faces(known_encodings, unknown_encoding)
    result = list(compress(known_names, matches))
    return result


if __name__ == "__main__":
    known_image = face_recognition.load_image_file("obama.jpeg")
    unknown_image = face_recognition.load_image_file("obama2.jpeg")
    unknown_image2 = face_recognition.load_image_file("krzych.jpeg")

    initialize_service()
    add_known_encoding(known_image, "Barrack Obama")
    detection = recognize(unknown_image)
    print(detection)
    assert detection == ["Barrack Obama"]
    detection = recognize(unknown_image2)
    print (detection)
    assert detection == []

    CLOUD.expose_service("face_recognition")
    CLOUD.save()
