import cv2
import numpy as np
import os
import sys

from mrcnn import utils
from mrcnn import model as modellib

import serial

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

arduino = serial.Serial('COM3',11520)

ROOT_DIR = os.path.abspath("./")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
sys.path.append(os.path.join(ROOT_DIR,"samples/coco/"))
import coco
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

model = modellib.MaskRCNN(
    mode="inference", model_dir=MODEL_DIR, config=config
)
model.load_weights(COCO_MODEL_PATH, by_name=True)
class_names = [
    'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]


def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors


colors = random_colors(len(class_names))
class_dict = {
    name: color for name, color in zip(class_names, colors)
}


def apply_mask(image, mask, color, alpha=0.5):
    """apply mask to image"""
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image


def display_instances(image, boxes, masks, ids, names, scores):
    """
        take the image and results and apply the mask, box, and Label
    """
    n_instances = boxes.shape[0]

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i in range(n_instances):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        color = class_dict[label]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]

        image = apply_mask(image, mask, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        image = cv2.putText(
            image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
        )

    return image


if __name__ == '__main__':
    """
        test everything
    """
    # Create data
    N = 500
    x = np.random.rand(N)
    y = np.random.rand(N)
    colors = (0,0,0)
    area = np.pi*3

    # Plot

    capture = cv2.VideoCapture(1)

    # these 2 lines can be removed if you dont have a 1080p camera.
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    while True:
        ret, frame = capture.read()
        results = model.detect([frame], verbose=0)
        r = results[0]
        frame = display_instances(
            frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
        Classid = r['class_ids']
        hatem = r['features']
        print(hatem)
        print(hatem.shape)
        X1 = hatem.transpose()
        print(X1)
        if X1.shape[1] >= 3:
            print('den')
            colors = (0,0,0)
            area = np.pi*3
            '''plt.scatter(X1[:, 0], X1[:, 1] ,s=area, c=colors, alpha=0.5)
            plt.title('Scatter plot pythonspot.com')
            plt.xlabel('x')
            plt.ylabel('y')
            #plt.show()
            #plt.savefig("mygraph.png")'''
            inertias = []
            distortions = []
            inertias = []
            mapping1 = {}
            mapping2 = {}
            K = range(1, 9)
            for k in K:
                # Building and fitting the model
                kmeanModel = KMeans(n_clusters=k).fit(X1)
                kmeanModel.fit(X1)
                distortions.append(sum(np.min(cdist(X1, kmeanModel.cluster_centers_,
                                                    'euclidean'), axis=1)) / X1.shape[0])
                inertias.append(kmeanModel.inertia_)
                mapping1[k] = sum(np.min(cdist(X1, kmeanModel.cluster_centers_,
                                               'euclidean'), axis=1)) / X1.shape[0]
                mapping2[k] = kmeanModel.inertia_
            plt.plot(K, distortions, 'bx-')
            plt.xlabel('Values of K')
            plt.ylabel('Distortion')
            plt.title('The Elbow Method using Distortion')
            plt.show()
            plt.savefig("elbowmethodgraph.png")
            kmeans = KMeans(n_clusters=2)
            kmeans.fit(X1)
            y_kmeans = kmeans.predict(X1)
            #plt.scatter(hatem[:, 0], hatem[:, 1], c=y_kmeans, s=50, cmap='viridis')
            centers = kmeans.cluster_centers_
            
       
        for i in range(len(Classid)):
            if Classid[i] == 77:
                arduino.write(str.encode('1'))
            if Classid[i] == 74:
                arduino.write(str.encode('2'))
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            arduino.write(str.encode('0'))
            break

    capture.release()
    cv2.destroyAllWindows()