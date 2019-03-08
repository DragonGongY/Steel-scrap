import os
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import visualize
import model as modellib
from model import log

import sqlite3

# conn = sqlite3.connect(server='{sql server}', driver='fgsb', uid='sa', pwd='qaz741852963')

from scrap import scrap

# Directory to save logs and trained model
MODEL_DIR = os.path.join("./", "logs")

config = scrap.ScrapConfig()
class InferenceConfig(config.__class__):
    # Run detection on one image at a time.
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    # IMAGE_MIN_DIM = 800
    # IMAGE_MAX_DIM = 1024

    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 320

config = InferenceConfig()
config.display()

# DEVICE = "/gpu:0"  # /gpu:0
DEVICE = "/cpu"

TEST_MODE = "inference"

def get_ax(rows=1, cols=1, size=16):
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

# Load validation dataset.
dataset = scrap.ScrapDataset()
dataset.load_scrap("datasets/scrap/", "train")

# Must call before using the dataset.
dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)
weights_path = "./mask_rcnn_scrap_0126.h5"

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

image_id = random.choice(dataset.image_ids)
image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
info = dataset.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                       dataset.image_reference(image_id)))

# Run object detection
results = model.detect([image], verbose=1)

# Display results
ax = get_ax(1)
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            dataset.class_names, r['scores'], ax=ax,
                            title="Predictions")
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)
