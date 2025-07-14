# import the necessary packages
from keras.applications import VGG16, ResNet50, imagenet_utils
from keras.preprocessing.image import img_to_array, load_img
from imutils import paths
import numpy as np
import argparse
import pickle
import imutils
import h5py
import cv2

# --- Parse arguments ---
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--db", required=True, help="path to HDF5 features database")
ap.add_argument("-i", "--dataset", required=True, help="path to input image folder")
ap.add_argument("-m", "--model", required=True, help="path to trained orientation model")
args = vars(ap.parse_args())

# --- Load label names from HDF5 ---
print("[INFO] loading label names...")
db = h5py.File(args["db"], "r")
labelNames = [int(l) for l in db["label_names"][:]]
db.close()

# --- Load random image paths ---
print("[INFO] sampling 10 images...")
imagePaths = list(paths.list_images(args["dataset"]))
imagePaths = np.random.choice(imagePaths, size=(10,), replace=False)

# --- Load VGG16 and ResNet50 models ---
print("[INFO] loading VGG16 and ResNet50 models...")
vgg = VGG16(weights="imagenet", include_top=False)
resnet = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# --- Load trained orientation classifier ---
print("[INFO] loading classifier model...")
model = pickle.loads(open(args["model"], "rb").read())

# --- Predict and correct orientation ---
for imagePath in imagePaths:
    # Load image via OpenCV
    orig = cv2.imread(imagePath)

    # Load and preprocess image for Keras
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # --- Extract features ---
    vgg_feat = vgg.predict(image)                # (1, 512, 7, 7)
    vgg_feat = vgg_feat.reshape((1, 512 * 7 * 7)) # → (1, 25088)
    resnet_feat = resnet.predict(image)          # → (1, 2048)

    # --- Combine features ---
    combined = np.hstack([vgg_feat, resnet_feat])  # → (1, 27136)

    # --- Predict orientation ---
    pred = model.predict(combined)
    angle = labelNames[pred[0]]
    print(f"[INFO] {imagePath} predicted angle: {angle}°")

    # --- Rotate image to correct orientation ---
    rotated = imutils.rotate_bound(orig, 360 - angle)

    # --- Show results ---
    cv2.imshow("Original", orig)
    cv2.imshow("Corrected", rotated)
    cv2.waitKey(0)
