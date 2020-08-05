import tensorflow as tf
import time

def read_and_decode_for_one_img(tfrecord_filename, batch_size = 1):
    
    filename_queue = tf.train.string_input_producer([tfrecord_filename], shuffle = False)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
            serialized_example,
            features = {
                    'img_raw':tf.FixedLenFeature([], tf.string),
                    'img_label':tf.FixedLenFeature([], tf.string),
                    'img_height':tf.FixedLenFeature([], tf.int64),
                    'img_width':tf.FixedLenFeature([], tf.int64)
                    }
            )

    width = tf.cast(features['img_height'], tf.int32)
    height = tf.cast(features['img_width'], tf.int32)
    
    img_cur = tf.cast(tf.reshape(tf.decode_raw(features['img_raw'], tf.uint8), [1, height, width, 3]), tf.float32) / 255.0
    img_clean_cur = tf.cast(tf.reshape(tf.decode_raw(features['img_label'], tf.uint8), [1, height, width, 3]), tf.float32) / 255.0

    # if(patch_height):
    #     _time = time.time()
    #     img_cur = tf.random_crop(img_cur, [1,patch_width, patch_height, 3], seed=_time)
    #     img_clean_cur = tf.random_crop(img_clean_cur, [1,patch_width, patch_height, 3], seed=_time)
    
    return img_cur, img_clean_cur
    
def read_and_decode(tfrecord_filename, patch_width, patch_height, batch_size):
    
    filename_queue = tf.train.string_input_producer([tfrecord_filename], shuffle = True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
            serialized_example,
            features = {
                    'img_raw':tf.FixedLenFeature([], tf.string),
                    'img_label':tf.FixedLenFeature([], tf.string),
                    'img_height':tf.FixedLenFeature([], tf.int64),
                    'img_width':tf.FixedLenFeature([], tf.int64)
                    }
            )
    width = tf.cast(features['img_height'], tf.int32)
    height = tf.cast(features['img_width'], tf.int32)
    
    img_cur = tf.cast(tf.reshape(tf.decode_raw(features['img_raw'], tf.uint8), [height, width, 3]), tf.float32) / 255.0
    img_clean_cur = tf.cast(tf.reshape(tf.decode_raw(features['img_label'], tf.uint8), [height, width, 3]), tf.float32) / 255.0
    
    _time = time.time()
    img_cur = tf.random_crop(img_cur, [patch_width, patch_height, 3], seed = _time)
    img_clean_cur = tf.random_crop(img_clean_cur, [patch_width, patch_height, 3], seed = _time)
    
    img_cur, img_clean_cur = tf.train.shuffle_batch([img_cur, img_clean_cur], batch_size = batch_size, capacity = 5000, 
                                                    min_after_dequeue = 1000)
    
    return img_cur, img_clean_cur    