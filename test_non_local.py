import os
import numpy as np
# from model import *
# from model_1 import *
# from model_2 import *
# from non_local_res_guide_2 import *
from non_local_res_guide_dilation import *
import tensorflow as tf
from matplotlib import image as mping
import cv2

################# Select GPU device ##################
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
######################################################

tf.reset_default_graph()

################# Network parameters ##################
num_feature = 16  # number of feature maps
num_channels = 3  # number of inputs's channels
repeat = 5  # num of recursive
# patch_height = 480  # training patch height
# patch_width = 720
patch_height = 960  # training patch height
patch_width = 720
# test_img_path = './TestData/test_a/rain'      # path of testing image
test_img_path = './TestData/image/data/'
derain_img_path = './TestData/image/results/'
# derain_img_path = './TestData/test_a/results/' # path to save result
load_model = True  # flag of load_model
total_parameters = 0  # num of paramters
######################################################

################# Model ##################
# checkpoint_dir = './checkpoint/Rain100H/' # try the pre-trianed model
# checkpoint_dir = './checkpoint/derain_model_1/'           # try your new model
# checkpoint_dir = './checkpoint/derain_model/'
checkpoint_dir = './checkpoint/dilation_non_local_model/'
################# Model ##################


if __name__ == '__main__':

    # rain_img = tf.placeholder(dtype = tf.float32, shape = (1, None, None, 3), name='rain_img')
    rain_img = tf.placeholder(dtype=tf.float32, shape=(1, patch_height, patch_width, 3), name='rain_img')
    _, _, _, _, result = derain_net(rain_img, num_channels, num_feature, num=repeat)
    # _, _, _, _, result,res = derain_net(rain_img, num_channels, num_feature, num=repeat)
    t_vars = tf.trainable_variables()
    for var in t_vars:
        shape = var.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("Total training params: %f" % (total_parameters))

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print('load_sucess')

        im_names = os.listdir(test_img_path)
        num = len(im_names)

        if not os.path.exists(derain_img_path):
            os.makedirs(derain_img_path)

        for i in range(num):

            rain = mping.imread(test_img_path + im_names[i])
            rain = cv2.resize(
                rain, (patch_width, patch_height), interpolation=cv2.INTER_LINEAR
            )
            rain = rain[:, :, 0:3]
            if np.max(rain) > 1:
                rain = rain / 255.0

            # H,W,C = np.shape(rain)
            # rain = np.reshape(rain, (1,H,W,C))
            rain = np.reshape(rain, (1, patch_height, patch_width, 3))
            derain = sess.run(result, feed_dict={rain_img: rain})
            # derain = sess.run(res, feed_dict={rain_img: rain})
            save_images_(derain, [1, 1], derain_img_path + im_names[i][:-8] + 'de' + im_names[i][-8:])
            print(im_names[i])

        sess.close()

