import tensorflow as tf
import os
import cv2
from PIL import Image

data_dir="/home/nigga/engine/data/cats"
images_ext=["jpg","jpeg","bmp","png"]

for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir,image_class)):
        image_path=os.path.join(data_dir,image_class,image)
        try:
            img=cv2.imread(image_path)
            tip=Image.open(image_path).format.lower()
            if tip not in images_ext:
                print("image not in ext list{}".format(image_path))
                os.remove(image_path)
        except Exception as e:
            print("essue with image {}".format(image_path))

data=tf.keras.utils.image_dataset_from_directory(data_dir)
data_iterator=data.as_numpy_iterator()
batch=data_iterator.next()

data=data.map(lambda x,y:(x/255,y))

train_size=int(len(data)*0.7)
val_size=int(len(data)*0.2)
test_size=int(len(data)*0.1)

train=data.take(train_size)
val=data.skip(train_size).take(val_size)
test=data.skip(train_size+val_size).take(test_size)

train_size=int(len(data)*0.7)
val_size=int(len(data)*0.2)
test_size=int(len(data)*0.1)

train=data.take(train_size)
val=data.skip(train_size).take(val_size)
test=data.skip(train_size+val_size).take(test_size)