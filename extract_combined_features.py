from keras.applications import VGG16, ResNet50
from keras.applications import imagenet_utils
from keras.applications.vgg16 import preprocess_input as vgg_preprocess
from keras.applications.resnet50 import preprocess_input as resnet_preprocess
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
from keras.layers import GlobalAveragePooling2D
from sklearn.preprocessing import LabelEncoder
from processor.hdf5_dataset_writer import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import argparse
import random
import os

# --- Argument parsing ---
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to input image dataset")
ap.add_argument("-o", "--output", required=True, help="Path to output HDF5 file")
ap.add_argument("-b", "--batch-size", type=int, default=32, help="Batch size")
ap.add_argument("-s", "--buffer-size", type=int, default=1000, help="Buffer size")
args = vars(ap.parse_args())

bs = args["batch_size"]

# --- Load & shuffle image paths ---
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
random.shuffle(imagePaths)

# --- Encode class labels ---
labels = [p.split(os.path.sep)[-2] for p in imagePaths]
le = LabelEncoder()
labels = le.fit_transform(labels)

# --- Load base models (without top), then add pooling ---
print("[INFO] loading VGG16 and ResNet50...")
vgg_base = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
resnet_base = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

vgg_model = Model(inputs=vgg_base.input, outputs=GlobalAveragePooling2D()(vgg_base.output))  # → 512-d
resnet_model = Model(inputs=resnet_base.input, outputs=GlobalAveragePooling2D()(resnet_base.output))  # → 2048-d

# --- Initialize HDF5 writer ---
feature_dim = 512 + 2048  # VGG + ResNet
dataset = HDF5DatasetWriter((len(imagePaths), feature_dim), args["output"], dataKey="features", bufSize=args["buffer_size"])
dataset.storeClassLabels(le.classes_)

# --- Progress bar ---
widgets = ["Extracting Features: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(imagePaths), widgets=widgets).start()

# --- Process images in batch ---
for i in np.arange(0, len(imagePaths), bs):
    batchPaths = imagePaths[i:i + bs]
    batchLabels = labels[i:i + bs]
    batchImages = []

    for imagePath in batchPaths:
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        batchImages.append(image)

    # Stack and preprocess for both models
    batchImages = np.vstack(batchImages)
    vgg_input = vgg_preprocess(batchImages.copy())
    resnet_input = resnet_preprocess(batchImages.copy())

    # Extract features
    vgg_features = vgg_model.predict(vgg_input, batch_size=bs)  # (bs, 512)
    resnet_features = resnet_model.predict(resnet_input, batch_size=bs)  # (bs, 2048)

    # Concatenate features
    features = np.hstack([vgg_features, resnet_features])  # (bs, 2560)

    # Write to HDF5
    dataset.add(features, batchLabels)
    pbar.update(i)

# --- Finish ---
dataset.close()
pbar.finish()
print("[INFO] Feature extraction completed and saved to HDF5.")
