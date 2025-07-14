from keras.applications import VGG16, ResNet50, imagenet_utils
from keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import LabelEncoder
from processor.hdf5_dataset_writer import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import argparse
import random
import os

# --- Parse arguments ---
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to input dataset")
ap.add_argument("-o", "--output", required=True, help="Path to output HDF5 file")
ap.add_argument("-b", "--batch-size", type=int, default=32, help="Batch size")
ap.add_argument("-s", "--buffer-size", type=int, default=1000, help="HDF5 buffer size")
args = vars(ap.parse_args())

# --- Parameters ---
bs = args["batch_size"]

# --- Load image paths & shuffle ---
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
random.shuffle(imagePaths)

# --- Encode labels ---
labels = [p.split(os.path.sep)[-2] for p in imagePaths]
le = LabelEncoder()
labels = le.fit_transform(labels)

# --- Load models ---
print("[INFO] loading networks...")
vgg = VGG16(weights="imagenet", include_top=False)
resnet = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# --- Init HDF5 writer ---
total_dim = 512 * 7 * 7 + 2048  # 25088 + 2048 = 27136
dataset = HDF5DatasetWriter((len(imagePaths), total_dim), args["output"], dataKey="features", bufSize=args["buffer_size"])
dataset.storeClassLabels(le.classes_)

# --- Progress bar ---
widgets = ["Extracting Features: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(imagePaths), widgets=widgets).start()

# --- Loop over image batches ---
for i in np.arange(0, len(imagePaths), bs):
    batchPaths = imagePaths[i:i + bs]
    batchLabels = labels[i:i + bs]
    batchImages = []

    for imagePath in batchPaths:
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)
        batchImages.append(image)

    batchImages = np.vstack(batchImages)

    # --- Extract VGG16 features ---
    vgg_features = vgg.predict(batchImages, batch_size=bs)
    vgg_features = vgg_features.reshape((vgg_features.shape[0], 512 * 7 * 7))

    # --- Extract ResNet50 features ---
    resnet_features = resnet.predict(batchImages, batch_size=bs)

    # --- Concatenate ---
    combined_features = np.hstack([vgg_features, resnet_features])

    # --- Save features ---
    dataset.add(combined_features, batchLabels)
    pbar.update(i)

# --- Done ---
dataset.close()
pbar.finish()
print("[INFO] feature extraction done.")
