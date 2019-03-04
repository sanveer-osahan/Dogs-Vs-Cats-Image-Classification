import cv2
import os
import shutil
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
from get_model import get_model


def create_test_data():
    testing_data = []
    for img in tqdm(os.listdir(RESULTS_DIR)):
        path = os.path.join(RESULTS_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img), img_num])
    np.save('test_data.npy', testing_data)
    return testing_data


LR = 1e-3  # Learning rate is 0.001
IMG_SIZE = 56
TEST_DIR = "./dogs_vs_cats_images_dataset/test"
MODEL_NAME = "dogs_vs_cats-{}-{}.model".format(LR, '6conv-basic')
RESULTS_DIR = "./results"
RESULTS_NUM = 50

if os.path.exists(RESULTS_DIR):
    shutil.rmtree(RESULTS_DIR)

os.mkdir(RESULTS_DIR)

# Pick random 20 images from TEST_DIR to RESULTS_DIR
for i in random.sample(range(1, 12500), RESULTS_NUM):
    shutil.copy(os.path.join(TEST_DIR, str(i)+".jpg"), RESULTS_DIR)

test_data = create_test_data()

model = get_model(MODEL_NAME)

for data in test_data:
    img_data = data[0]
    img_name = data[1]

    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)

    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 1:
        label = 'Dog_' + str(int(model_out[1] * 100))
    else:
        label = 'Cat_' + str(int(model_out[0] * 100))

    os.rename(os.path.join(RESULTS_DIR, str(img_name)+".jpg"),
              os.path.join(RESULTS_DIR, str(img_name)+"_"+label+".jpg"))

'''def create_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img), img_num])
    np.save('test_data.npy', testing_data)
    return testing_data


LR = 1e-3  # Learning rate is 0.001
IMG_SIZE = 56
TEST_DIR = './dogs_vs_cats_images_dataset/test'
MODEL_NAME = 'dogs_vs_cats-{}-{}.model'.format(LR, '6conv-basic')

if os.path.exists("test_data.npy"):
    test_data = np.load("test_data.npy")
    print("Loaded testing data")
else:
    print("Creating test data....")
    test_data = create_test_data()

model = get_model(MODEL_NAME)
fig = plt.figure()

for num, data in enumerate(test_data[:12]):
    # cat: [1,0]
    # dog: [0,1]

    img_num = data[1]
    img_data = data[0]

    y = fig.add_subplot(3, 4, num + 1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
    # model_out = model.predict([data])[0]
    model_out = model.predict([data])[0]

    print(model_out)

    if np.argmax(model_out) == 1:
        str_label = 'Dog ' + str(int(model_out[1]*100)) + "%"
    else:
        str_label = 'Cat ' + str(int(model_out[0]*100)) + "%"

    y.imshow(orig, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()
'''