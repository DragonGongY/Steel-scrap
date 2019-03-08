import os
import utils
import tensorflow as tf
import matplotlib.pyplot as plt
import visualize
import model as modellib
import skimage
from skimage import color, transform, io
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
# config.display()

def get_ax(rows=1, cols=1, size=16):
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

model = modellib.MaskRCNN(mode="inference",
                          model_dir=MODEL_DIR, config=config)

# Set classes including BG.
class_names = ['BG', '2', '4', '6', '8', '10', '20', 'bottle', 'car', 'none']

# The trained weights path.
weights_path = "./mask_rcnn_scrap_0126.h5"

# Load weights
# print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

def load_rec(image):
    # image = skimage.io.imread(image)
    # img = color.rgb2gray(image)
    # img = transform.resize(img, (1920,1080))
    # Run object detection
    results = model.detect([image], verbose=1)
    # Display results
    ax = get_ax(1)
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'], ax=ax, title='Predictions')

if __name__ == "__main__":
    load_rec('./datasets/scrap/val/201881501347.jpg')