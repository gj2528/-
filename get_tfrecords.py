import tensorflow as tf
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))
    
def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

deep_joint_rain_train_data = './TrainData/' ## path of rain img
deep_joint_rain_train_data_names = os.listdir(deep_joint_rain_train_data + 'rain_image/')  ## names of rain img

deep_joint_clean_train_data_names = os.listdir(deep_joint_rain_train_data + 'clean_image/')  ## names of rain img



num = len(deep_joint_rain_train_data_names)
count = 0



writer_train_data = tf.python_io.TFRecordWriter(deep_joint_rain_train_data + 'Rain_train.tfrecords') ## name of tfrecords
for i in range(int(0.9*num)):

    im_name = deep_joint_rain_train_data_names[i]
    name = im_name.split('/')[-1].split('.')[0]
    clean_name = deep_joint_clean_train_data_names[i]
    
    rain_img = plt.imread(deep_joint_rain_train_data + 'rain_image/' + im_name) * 255.0
    rain_img_raw = rain_img.astype(np.uint8)
    img_raw = rain_img_raw.tobytes()
    

    clean_img = plt.imread(deep_joint_rain_train_data + 'clean_image/' + clean_name) * 255.0 ## path of gt img
    clean_img_raw = clean_img.astype(np.uint8)
    img_label = clean_img_raw.tobytes()
    
    M, N, C = np.shape(rain_img)
    example = tf.train.Example(features = tf.train.Features(
            feature = {
                    "img_raw": _bytes_feature(img_raw),
                    "img_label": _bytes_feature(img_label),
                    "img_height": _int64_feature(N),
                    "img_width": _int64_feature(M)}))

    writer_train_data.write(example.SerializeToString())
    count += 1
    print("img_name:%s, num_of_img:%d/%d" %(name, count, len(deep_joint_rain_train_data_names)))
writer_train_data.close()



writer_val_data = tf.python_io.TFRecordWriter(deep_joint_rain_train_data + 'Rain_val.tfrecords') ## name of tfrecords
for i in range(int(0.9*num),num):

    im_name = deep_joint_rain_train_data_names[i]
    name = im_name.split('/')[-1].split('.')[0]
    clean_name = deep_joint_clean_train_data_names[i]
    
    rain_img = plt.imread(deep_joint_rain_train_data + 'rain_image/' + im_name) * 255.0
    rain_img_raw = rain_img.astype(np.uint8)
    img_raw = rain_img_raw.tobytes()
    

    clean_img = plt.imread(deep_joint_rain_train_data + 'clean_image/' + clean_name) * 255.0 ## path of gt img
    clean_img_raw = clean_img.astype(np.uint8)
    img_label = clean_img_raw.tobytes()
    
    M, N, C = np.shape(rain_img)
    example = tf.train.Example(features = tf.train.Features(
            feature = {
                    "img_raw": _bytes_feature(img_raw),
                    "img_label": _bytes_feature(img_label),
                    "img_height": _int64_feature(N),
                    "img_width": _int64_feature(M)}))

    writer_val_data.write(example.SerializeToString())
    count += 1
    print("img_name:%s, num_of_img:%d/%d" %(name, count, len(deep_joint_rain_train_data_names)))
writer_val_data.close()
