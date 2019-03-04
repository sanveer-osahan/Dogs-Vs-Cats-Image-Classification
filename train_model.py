import os
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from random import shuffle
from get_model import get_model


def label_img(img):
    word_label = img.split('.')[0]
    if word_label == 'cat':
        return [1, 0]
    elif word_label == 'dog':
        return [0, 1]


def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


TRAIN_DIR = "./dogs_vs_cats_images_dataset/train"
IMG_SIZE = 56
LR = 1e-3  # Learning rate is 0.001

MODEL_NAME = "dogs_vs_cats-{}-{}.model".format(LR, '6conv-basic')

if os.path.exists("train_data.npy"):
    train_data = np.load("train_data.npy")
    print("Loaded training data")
else:
    print("Creating training data....")
    train_data = create_train_data()


tf.reset_default_graph()

model = get_model(MODEL_NAME)

train = train_data[:-500]
test = train_data[-500:]

train_X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
train_Y = [i[1] for i in train]

test_X = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_Y = [i[1] for i in test]

model.fit({'input': train_X}, {'targets': train_Y}, n_epoch=2, validation_set=({'input': test_X}, {'targets': test_Y}),
          snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME)
