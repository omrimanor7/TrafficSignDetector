# Import Harvester:
from harvesters.core import Harvester

import keras
from tensorflow.keras.models import load_model as load_recognition_model

import keras_retinanet
from keras_retinanet import models
from keras_retinanet.models import load_model as load_detection_model
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from skimage import transform, exposure

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import pandas as pd
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

THRESH_SCORE = 0.5

detection_model_path = "../Models/detection_model.h5"
recognition_model_path = "../Models/recognition_model.h5"

CLASSES_FILE = '../Annotations/classes.csv'
ANNOTATIONS_FILE = '../Annotations/annotations.csv'


global detection_model
global recognition_model
global label_to_names


def run_camera():
    # Create a Harvester object:
    h = Harvester()

    # Load a GenTL Producer; you can load many more if you want to:   ##find producer
    h.add_file('C:\Program Files\Allied Vision\Vimba_4.2\VimbaGigETL\Bin\Win64\VimbaGigETL.cti')

    # Enumerate the available devices that GenTL Producers can handle:
    h.update()
    print(h.device_info_list)

    # Select a target device and create an ImageAcquire object that
    # controls the device:
    ia = h.create_image_acquirer(0)

    # Configure the target device; it looks very small but this is just
    # for demonstration:
    # ia.remote_device.node_map.Width.value = 1936
    # ia.remote_device.node_map.Height.value = 1216
    # ia.remote_device.node_map.PixelFormat.value = 'RGB8Packed'

    # Allow the ImageAcquire object to start image acquisition:
    ia.start_acquisition()

    # We are going to fetch a buffer filled up with an image:
    # Note that you'll have to queue the buffer back to the
    # ImageAcquire object once you consumed the buffer; the
    # with statement takes care of it on behalf of you:
    with ia.fetch_buffer() as buffer:
        # Let's create an alias of the 2D image component; it can be
        # lengthy which is not good for typing. In addition, note that
        # a 3D camera can give you 2 or more components:
        component = buffer.payload.components[0]

        # Reshape the NumPy array into a 2D array:
        image_rgb = component.data.reshape(
            component.height, component.width, 3
        )

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # img_inference runs both models
        processing_time, boxes, scores, labels = img_inference(image_bgr)

    # Stop the ImageAcquier object acquiring images:
    ia.stop_acquisition()

    # We're going to leave here shortly:
    ia.destroy()
    h.reset()


def run_detection_model(image):
    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    start = time.time()
    boxes, scores, labels = detection_model.predict_on_batch(np.expand_dims(image, axis=0))
    processing_time = time.time() - start
    print("processing time for detection model: ", processing_time)

    # correct for image scale
    boxes /= scale

    return boxes[0], scores[0], labels[0]


def classify_img(crop_im):
    image = transform.resize(crop_im, (64, 64))
    image = exposure.equalize_adapthist(image, clip_limit=0.1)

    # preprocess the image by scaling it to the range [0, 1]
    image = image.astype("float32")  # / 255.0
    image = np.expand_dims(image, axis=0)

    # make predictions using the traffic sign recognizer CNN
    preds = recognition_model.predict(image)
    j = preds.argmax(axis=1)[0]
    score = preds[0][j]
    label = label_to_names[j]

    return label, j, score


def run_recognition_model(image, boxes, scores):
    recognition_labels = list()
    class_ids = list()
    reco_scores = list()  # scores from recognition model

    start = time.time()

    for i in range(len(boxes)):
        if scores[i] < THRESH_SCORE:
            break
        start_x, start_y, end_x, end_y = np.array(boxes[i]).astype(int)
        cropped_image = image[start_y:end_y, start_x:end_x, :]
        label, class_id, score = classify_img(cropped_image)
        recognition_labels.append(label)
        class_ids.append(class_id)
        reco_scores.append(score)

    processing_time = time.time() - start
    print("processing time for recognition model: ", processing_time)

    return class_ids, reco_scores, recognition_labels


def draw_text(img, text,
              font=cv2.FONT_HERSHEY_PLAIN,
              pos=(0, 0),
              font_scale=1,
              font_thickness=2,
              text_color=(255, 255, 255),
              text_color_bg=(0, 0, 0)
              ):
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size


def visualize_detections(draw, boxes, reco_scores, class_ids):
    start = time.time()
    for box, score, label_id in zip(boxes, reco_scores, class_ids):

        # print(label)

        # scores are sorted so we can break
        if score < THRESH_SCORE:
            break

        color = label_color(label_id)

        b = box.astype(int)
        draw_box(draw, b, color=color)

        caption = "{} {:.3f}".format(label_to_names[label_id], score)
        draw_text(draw, caption, pos=(b[0], b[1] - 10))

    processing_time = time.time() - start
    print("processing time to visualize: ", processing_time)

    # show the output image_rgb
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(draw)
    plt.show(block=False)


def img_inference(image):
    # This functions runs both models (recognition and detection) and returns the final result
    start = time.time()

    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # Detection model
    boxes, scores, labels = run_detection_model(image)

    # Recognition model: (runs over the cropped images)
    class_ids, reco_scores, recognition_labels = run_recognition_model(draw, boxes, scores)

    # Visualize detections
    visualize_detections(draw, boxes, reco_scores, class_ids)

    processing_time = time.time() - start
    print("processing time to infer img: ", processing_time)

    return processing_time, boxes, reco_scores, class_ids


def get_session():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.compat.v1.Session(config=config)


def init():

    global detection_model
    global recognition_model
    global label_to_names

    # set the modified tf session as backend in keras
    tf.compat.v1.keras.backend.set_session(get_session())

    # load retinanet model
    print("[INFO] loading detection model...")
    detection_model = load_detection_model(detection_model_path, backbone_name='resnet50')
    detection_model = models.convert_model(detection_model)

    # load label to names mapping for visualization purposes
    labels_to_names = pd.read_csv(CLASSES_FILE, header=None).T.loc[0].to_dict()

    # load the traffic sign recognizer model
    print("[INFO] loading recognition model...")
    recognition_model = load_recognition_model(recognition_model_path)

    # load the label names
    label_to_names = open(CLASSES_FILE).read().strip().split("\n")
    label_to_names = [l.split(",")[0] for l in label_to_names]


if __name__ == "__main__":
    init()
    run_camera()
