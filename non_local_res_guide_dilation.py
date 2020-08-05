import tensorflow as tf
import numpy as np
import scipy.misc
import scipy.io
from non_local_block import non_local_block


from cnn_basenet import CNNBaseModel as cb

DEFAULT_PADDING = 'SAME'


def validate_padding(padding):
    assert padding in ('SAME', 'VALID')


def make_var(name, shape, initializer=None, trainable=True, regularizer=None):
    return tf.get_variable(name, shape, initializer=initializer, trainable=trainable, regularizer=regularizer)


def l2_regularizer(weight_decay=0.0005, scope=None):
    def regularizer(tensor):
        with tf.name_scope(scope, default_name='l2_regularizer', values=[tensor]):
            l2_weight = tf.convert_to_tensor(weight_decay,
                                             dtype=tensor.dtype.base_dtype,
                                             name='weight_decay')
            return tf.multiply(l2_weight, tf.nn.l2_loss(tensor), name='value')

    return regularizer


def max_pool(input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
    validate_padding(padding)
    return tf.nn.max_pool(input,
                          ksize=[1, k_h, k_w, 1],
                          strides=[1, s_h, s_w, 1],
                          padding=padding,
                          name=name)



# def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer()):
#     regularizer = tf.contrib.layers.l2_regularizer(scale=1e-4)
#
#     new_variables = tf.get_variable(name=name + "conv", shape=shape, initializer=initializer,
#                                     regularizer=regularizer)
#     return new_variables



def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    return img


def save_images_(images, size, image_path):
    images = images * 255.0
    images[np.where(images < 0)] = 0.
    images[np.where(images > 255)] = 255.
    images = images.astype(np.uint8)
    return scipy.misc.imsave(image_path, merge(images, size))


def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=tf.float32))
    return numerator / denominator


def _tf_fspecial_gauss(size, sigma):
    """ Function to mimic the 'fspecial' gaussian MATLAB functino
    """
    x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1: size // 2 + 1]
    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x ** 2 + y ** 2) / (2.0 * sigma * 2)))
    return g / tf.reduce_sum(g)


def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 1
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    mu1 = tf.nn.conv2d(img1, window, strides=[1, 1, 1, 1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1, 1, 1, 1], padding='VALID')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = tf.nn.conv2d(img1 * img1, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_mu2
    if cs_map:
        value = (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)),
                 (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if mean_metric:
        value = tf.reduce_mean(value)
    return value


def image_to_4d(image):
    image = tf.expand_dims(image, 0)
    image = tf.expand_dims(image, -1)
    return image


def loss_ssim(img1, img2, batchsize, c_dims):
    ssim_value_sum = 0
    for i in range(batchsize):
        for j in range(c_dims):
            img1_tmp = img1[i, :, :, j]
            img1_tmp = image_to_4d(img1_tmp)
            img2_tmp = img2[i, :, :, j]
            img2_tmp = image_to_4d(img2_tmp)
            ssim_value_tmp = tf_ssim(img1_tmp, img2_tmp)
            ssim_value_sum += ssim_value_tmp
    ssim_value_ave = ssim_value_sum / (batchsize * c_dims)
    return log10(1.0 / (ssim_value_ave + 1e-4))


def recursive_block1(rain_img, c_dim=3, output_channel=16, recursive_num=3, index=1, stride=1, is_train=True):
    with tf.variable_scope("rec_block_%d" % index):
        print('block1')
        conv1_1 = cb.conv2d(rain_img, output_channel, kernel_size=3, padding='SAME', stride=1, use_bias=True, name='conv1_1')
        conv1_1 = cb.lrelu(conv1_1, name='relu_1')
        dia_conv1 = cb.dilation_conv(input_tensor=conv1_1, k_size=3, out_dims=output_channel, rate=2,
                                       padding='SAME', use_bias=False, name='dia_conv_1')
        relu_7 = cb.lrelu(dia_conv1, name='relu_7')

        dia_conv2 = cb.dilation_conv(input_tensor=relu_7, k_size=3, out_dims=output_channel, rate=4,
                                       padding='SAME', use_bias=False, name='dia_conv_2')
        relu_8 = cb.lrelu(dia_conv2, name='relu_8')

        dia_conv3 = cb.dilation_conv(input_tensor=relu_8, k_size=3, out_dims=output_channel, rate=8,
                                       padding='SAME', use_bias=False, name='dia_conv_3')
        relu_9 = cb.lrelu(dia_conv3, name='relu_9')

        dia_conv4 = cb.dilation_conv(input_tensor=relu_9, k_size=3, out_dims=output_channel, rate=16,
                                       padding='SAME', use_bias=False, name='dia_conv_4')
        relu_10 = cb.lrelu(dia_conv4, name='relu_10')
        print('non_local_block start', conv1_1.shape)
        conv1_1 = non_local_block(relu_10)
        # print('non_local_block End',conv1_1.shape)
        # print(conv1_1.shape, 'conv1_1')
        conv_temp = conv1_1
        for i in range(recursive_num):
            conv_temp = cb.conv2d(conv_temp, output_channel, kernel_size=3, padding='SAME', stride=1, use_bias=True, name='conv_temp_1_%d'%(i+1))
            conv_temp = cb.lrelu(conv_temp, name='relu_2_%d'%(i+1))
            print(conv_temp.shape, 'conv_temp')
            conv_temp = cb.conv2d(conv_temp, output_channel, kernel_size=3, padding='SAME', stride=1, use_bias=True, name='conv_temp_2_%d'%(i+1))
            conv_temp = cb.lrelu(conv_temp, name='relu_3_%d'%(i+1))
            print(conv_temp.shape, 'conv_temp')
        conv1_2 = conv_temp
        res = cb.conv2d(conv1_2, c_dim, kernel_size=3, padding='SAME', stride=1, use_bias=True, name='res')
        res = cb.lrelu(res, name='relu_4')
        print(res.shape, 'res')
        return rain_img + res, res


def recursive_block2(rain_img, input_, res, c_dim=3, output_channel=16, recursive_num=3, index=2, stride=1,
                     is_train=True):
    with tf.variable_scope("rec_block_%d" % index):
        print('block2')
        conv1_1 = cb.conv2d(input_, output_channel, kernel_size=3, padding='SAME', stride=1, use_bias=True, name='conv1_1')
        conv1_1 = cb.lrelu(conv1_1, name='relu_1')
        dia_conv1 = cb.dilation_conv(input_tensor=conv1_1, k_size=3, out_dims=output_channel, rate=2,
                                     padding='SAME', use_bias=False, name='dia_conv_1')
        relu_7 = cb.lrelu(dia_conv1, name='relu_7')

        dia_conv2 = cb.dilation_conv(input_tensor=relu_7, k_size=3, out_dims=output_channel, rate=4,
                                     padding='SAME', use_bias=False, name='dia_conv_2')
        relu_8 = cb.lrelu(dia_conv2, name='relu_8')

        dia_conv3 = cb.dilation_conv(input_tensor=relu_8, k_size=3, out_dims=output_channel, rate=8,
                                     padding='SAME', use_bias=False, name='dia_conv_3')
        relu_9 = cb.lrelu(dia_conv3, name='relu_9')

        dia_conv4 = cb.dilation_conv(input_tensor=relu_9, k_size=3, out_dims=output_channel, rate=16,
                                     padding='SAME', use_bias=False, name='dia_conv_4')
        relu_10 = cb.lrelu(dia_conv4, name='relu_10')
        print('non_local_block start', conv1_1.shape)
        conv1_1 = non_local_block(relu_10)
        # print('non_local_block End', conv1_1.shape)
        conv_temp = tf.concat([conv1_1, res], 3, name='concat1')
        for i in range(recursive_num):
            conv_temp = cb.conv2d(conv_temp, output_channel, kernel_size=3, padding='SAME', stride=1, use_bias=True,
                                  name='conv_temp_1_%d'%(i+1))
            conv_temp = cb.lrelu(conv_temp, name='relu_2_%d'%(i+1))
            maps1 = conv_temp
            conv_temp = tf.concat([conv_temp, res], 3)
            conv_temp = cb.conv2d(conv_temp, output_channel, kernel_size=3, padding='SAME', stride=1, use_bias=True,
                                  name='conv_temp_2_%d'%(i+1))
            conv_temp = cb.lrelu(conv_temp, name='relu_3_%d'%(i+1))
            maps2 = conv_temp
            conv_temp = tf.concat([conv_temp, res], 3)
        conv1_2 = conv_temp
        res = cb.conv2d(conv1_2, c_dim, kernel_size=3, padding='SAME', stride=1, use_bias=True, name='res')
        res = cb.lrelu(res, name='relu_4')
        return rain_img + res, res, maps1, maps2


def recursive_block3(rain_img, input_, res1, res2, c_dim, output_channel=16, recursive_num=3, index=3, stride=1,
                     is_train=True):
    with tf.variable_scope("rec_block_%d" % index):
        print('block3')
        conv1_1 = cb.conv2d(input_, output_channel, kernel_size=3, padding='SAME', stride=1, use_bias=True,
                            name='conv1_1')
        conv1_1 = cb.lrelu(conv1_1, name='relu_1')
        dia_conv1 = cb.dilation_conv(input_tensor=conv1_1, k_size=3, out_dims=output_channel, rate=2,
                                     padding='SAME', use_bias=False, name='dia_conv_1')
        relu_7 = cb.lrelu(dia_conv1, name='relu_7')

        dia_conv2 = cb.dilation_conv(input_tensor=relu_7, k_size=3, out_dims=output_channel, rate=4,
                                     padding='SAME', use_bias=False, name='dia_conv_2')
        relu_8 = cb.lrelu(dia_conv2, name='relu_8')

        dia_conv3 = cb.dilation_conv(input_tensor=relu_8, k_size=3, out_dims=output_channel, rate=8,
                                     padding='SAME', use_bias=False, name='dia_conv_3')
        relu_9 = cb.lrelu(dia_conv3, name='relu_9')

        dia_conv4 = cb.dilation_conv(input_tensor=relu_9, k_size=3, out_dims=output_channel, rate=16,
                                     padding='SAME', use_bias=False, name='dia_conv_4')
        relu_10 = cb.lrelu(dia_conv4, name='relu_10')
        print('non_local_block start', conv1_1.shape)
        conv1_1 = non_local_block(relu_10)
        # print('non_local_block End', conv1_1.shape)
        conv_temp = tf.concat([conv1_1, res1, res2], 3)
        for i in range(recursive_num):
            conv_temp = cb.conv2d(conv_temp, output_channel, kernel_size=3, padding='SAME', stride=1, use_bias=True,
                                  name='conv_temp_1_%d'%(i+1))
            conv_temp = cb.lrelu(conv_temp, name='relu_2_%d'%(i+1))
            conv_temp = tf.concat([conv_temp, res1, res2], 3)
            conv_temp = cb.conv2d(conv_temp, output_channel, kernel_size=3, padding='SAME', stride=1, use_bias=True,
                                  name='conv_temp_2_%d'%(i+1))
            conv_temp = cb.lrelu(conv_temp, name='relu_3_%d'%(i+1))
            conv_temp = tf.concat([conv_temp, res1, res2], 3)

        conv1_2 = conv_temp
        res = cb.conv2d(conv1_2, c_dim, kernel_size=3, padding='SAME', stride=1, use_bias=True, name='res')
        res = cb.lrelu(res, name='relu_4')
        return rain_img + res, res


def recursive_block4(rain_img, input_, res1, res2, res3, c_dim, output_channel=16, recursive_num=3, index=4, stride=1,
                     is_train=True):
    with tf.variable_scope("rec_block_%d" % index):
        print('block4')
        conv1_1 = cb.conv2d(input_, output_channel, kernel_size=3, padding='SAME', stride=1, use_bias=True,
                            name='conv1_1')
        conv1_1 = cb.lrelu(conv1_1, name='relu_1')
        dia_conv1 = cb.dilation_conv(input_tensor=conv1_1, k_size=3, out_dims=output_channel, rate=2,
                                     padding='SAME', use_bias=False, name='dia_conv_1')
        relu_7 = cb.lrelu(dia_conv1, name='relu_7')

        dia_conv2 = cb.dilation_conv(input_tensor=relu_7, k_size=3, out_dims=output_channel, rate=4,
                                     padding='SAME', use_bias=False, name='dia_conv_2')
        relu_8 = cb.lrelu(dia_conv2, name='relu_8')

        dia_conv3 = cb.dilation_conv(input_tensor=relu_8, k_size=3, out_dims=output_channel, rate=8,
                                     padding='SAME', use_bias=False, name='dia_conv_3')
        relu_9 = cb.lrelu(dia_conv3, name='relu_9')

        dia_conv4 = cb.dilation_conv(input_tensor=relu_9, k_size=3, out_dims=output_channel, rate=16,
                                     padding='SAME', use_bias=False, name='dia_conv_4')
        relu_10 = cb.lrelu(dia_conv4, name='relu_10')
        print('non_local_block start', conv1_1.shape)
        conv1_1 = non_local_block(relu_10)
        # print('non_local_block End',conv1_1.shape)
        conv_temp = tf.concat([conv1_1, res1, res2, res3], 3)
        for i in range(recursive_num):
            conv_temp = cb.conv2d(conv_temp, output_channel, kernel_size=3, padding='SAME', stride=1, use_bias=True,
                                  name='conv_temp_1_%d'%(i+1))
            conv_temp = cb.lrelu(conv_temp, name='relu_2_%d'%(i+1))
            conv_temp = tf.concat([conv_temp, res1, res2, res3], 3)
            conv_temp = cb.conv2d(conv_temp, output_channel, kernel_size=3, padding='SAME', stride=1, use_bias=True,
                                  name='conv_temp_2_%d'%(i+1))
            conv_temp = cb.lrelu(conv_temp, name='relu_3_%d'%(i+1))
            conv_temp = tf.concat([conv_temp, res1, res2, res3], 3)

        conv1_2 = conv_temp
        res = cb.conv2d(conv1_2, c_dim, kernel_size=3, padding='SAME', stride=1, use_bias=True, name='res')
        res = cb.lrelu(res, name='relu_4')
        return rain_img + res, res


def recursive_block5(rain_img, input_, res1, res2, res3, res4, c_dim, output_channel=16, recursive_num=3, index=5,
                     stride=1, is_train=True):
    with tf.variable_scope("rec_block_%d" % index):
        print('block5')
        conv1_1 = cb.conv2d(input_, output_channel, kernel_size=3, padding='SAME', stride=1, use_bias=True,
                            name='conv1_1')
        conv1_1 = cb.lrelu(conv1_1, name='relu_1')
        dia_conv1 = cb.dilation_conv(input_tensor=conv1_1, k_size=3, out_dims=output_channel, rate=2,
                                     padding='SAME', use_bias=False, name='dia_conv_1')
        relu_7 = cb.lrelu(dia_conv1, name='relu_7')

        dia_conv2 = cb.dilation_conv(input_tensor=relu_7, k_size=3, out_dims=output_channel, rate=4,
                                     padding='SAME', use_bias=False, name='dia_conv_2')
        relu_8 = cb.lrelu(dia_conv2, name='relu_8')

        dia_conv3 = cb.dilation_conv(input_tensor=relu_8, k_size=3, out_dims=output_channel, rate=8,
                                     padding='SAME', use_bias=False, name='dia_conv_3')
        relu_9 = cb.lrelu(dia_conv3, name='relu_9')

        dia_conv4 = cb.dilation_conv(input_tensor=relu_9, k_size=3, out_dims=output_channel, rate=16,
                                     padding='SAME', use_bias=False, name='dia_conv_4')
        relu_10 = cb.lrelu(dia_conv4, name='relu_10')
        print('non_local_block start', conv1_1.shape)
        conv1_1 = non_local_block(relu_10)
        # print('non_local_block End', conv1_1.shape)
        conv_temp = tf.concat([conv1_1, res1, res2, res3, res4], 3)
        for i in range(recursive_num):
            conv_temp = cb.conv2d(conv_temp, output_channel, kernel_size=3, padding='SAME', stride=1, use_bias=True,
                                  name='conv_temp_1_%d'%(i+1))
            conv_temp = cb.lrelu(conv_temp, name='relu_2_%d'%(i+1))
            conv_temp = tf.concat([conv_temp, res1, res2, res3, res4], 3)
            conv_temp = cb.conv2d(conv_temp, output_channel, kernel_size=3, padding='SAME', stride=1, use_bias=True,
                                  name='conv_temp_2_%d'%(i+1))
            conv_temp = cb.lrelu(conv_temp, name='relu_3_%d'%(i+1))
            conv_temp = tf.concat([conv_temp, res1, res2, res3, res4], 3)

        conv1_2 = conv_temp
        res = cb.conv2d(conv1_2, c_dim, kernel_size=3, padding='SAME', stride=1, use_bias=True, name='res')
        res = cb.lrelu(res, name='relu_4')
        return rain_img + res, res



def derain_net(rain_img, c_dim, out_channel, num=5, is_train=True, reuse=False):
    with tf.variable_scope('recursive_net2'):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        #        result = gauss_layer(input_, c_dim, is_train = is_train)
        # rain_img = attention_block(rain_img, 3)
        output1, res1 = recursive_block1(rain_img, c_dim, out_channel, recursive_num=num)
        output2, res2, maps1, maps2 = recursive_block2(rain_img, output1, res1, c_dim, out_channel, recursive_num=num)
        output3, res3 = recursive_block3(rain_img, output2, res1, res2, c_dim, out_channel, recursive_num=num)
        output4, res4 = recursive_block4(rain_img, output3, res1, res2, res3, c_dim, out_channel, recursive_num=num)
        output5, res5 = recursive_block5(rain_img, output4, res1, res2, res3, res4, c_dim, out_channel,
                                         recursive_num=num)
    #        output = conv(tf.concat([output1, output2, output3, output4, output5], 3, name = 'concat'), 3, 3, 3, 1, 1, name = 'output')

    return output1, output2, output3, output4, output5

# if __name__ == '__main__':
#     input_image = tf.placeholder(dtype=tf.float32, shape=[1, 256, 256, 3])
#     net = derain_net(input_image, 3, 16, reuse = True)
#     for vv in tf.trainable_variables():
#         print(vv.name)
#     var_list = tf.trainable_variables()
#     print("Total parameters' number: %d"
#           % (np.sum([np.prod(v.get_shape().as_list()) for v in var_list])))
