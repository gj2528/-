import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr


# derain_img_path = './TestData/test_b/data/131_rain.jpg'      # path of testing image
derain_img_path = './TestData/test_b/test_b_dilation_non_local_result/131_derain2.jpg' # path to save result
# derain_img_path = './TestData/non_local_2_results/39_rain.jpg'
label_img_path = './TestData/test_b/gt/131_clean.jpg'
H = 480
W = 720

# rain = mping.imread(test_img_path + im_names[i])
#             rain = rain[:,:,0:3]
#             if np.max(rain) > 1:
#                rain = rain/255.0
# H,W,C = np.shape(rain)
derain_image_vis = None
if derain_img_path is not None:
    derain_image = cv2.imread(derain_img_path, cv2.IMREAD_COLOR)
    derain_image_vis = cv2.resize(
        derain_image, (W, H), interpolation=cv2.INTER_LINEAR
    )

label_image_vis = None
if label_img_path is not None:
    label_image = cv2.imread(label_img_path, cv2.IMREAD_COLOR)
    label_image_vis = cv2.resize(
        label_image, (W, H), interpolation=cv2.INTER_LINEAR
    )

if label_img_path is not None:
    label_image_vis_gray = cv2.cvtColor(label_image_vis, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    output_image_gray = cv2.cvtColor(derain_image_vis, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    psnr = compare_psnr(label_image_vis_gray, output_image_gray)
    ssim = compare_ssim(label_image_vis_gray, output_image_gray)

    print('SSIM: {:.5f}'.format(ssim))
    print('PSNR: {:.5f}'.format(psnr))