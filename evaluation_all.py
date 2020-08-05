import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
import os


# test_img_path = './TestData/test_b/data/'      # path of testing image
# derain_img_path = './TestData/test_b/results/' # path to save result
# derain_img_path = './TestData/test_b/non_local_1_results/'
derain_img_path = './TestData/image/results/'
label_img_path = './TestData/image/gt/'
H = 960
W = 720

# rain = mping.imread(test_img_path + im_names[i])
#             rain = rain[:,:,0:3]
#             if np.max(rain) > 1:
#                rain = rain/255.0
# H,W,C = np.shape(rain)

im_names = os.listdir(derain_img_path)
num = len(im_names)
psnr_all = 0
ssim_all = 0
for i in range(num):
    derain_image_vis = None
    if derain_img_path is not None:
        derain_image = cv2.imread(derain_img_path + im_names[i], cv2.IMREAD_COLOR)
        derain_image_vis = cv2.resize(
            derain_image, (W, H), interpolation=cv2.INTER_LINEAR
        )

    label_image_vis = None
    if label_img_path is not None:
        label_image = cv2.imread(label_img_path + im_names[i].split('_')[0] + '_clean.jpg', cv2.IMREAD_COLOR)
        label_image_vis = cv2.resize(
            label_image, (W, H), interpolation=cv2.INTER_LINEAR
        )

    if label_img_path is not None:
        label_image_vis_gray = cv2.cvtColor(label_image_vis, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
        output_image_gray = cv2.cvtColor(derain_image_vis, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
        psnr = compare_psnr(label_image_vis_gray, output_image_gray)
        ssim = compare_ssim(label_image_vis_gray, output_image_gray)
        print(im_names[i])
        print('SSIM: {:.5f}'.format(ssim))
        print('PSNR: {:.5f}'.format(psnr))

        psnr_all = psnr_all + psnr
        ssim_all = ssim_all + ssim
avg_psnr = psnr_all/num
avg_ssim = ssim_all/num
print('avg_SSIM: {:.5f}'.format(avg_ssim))
print('avg_PSNR: {:.5f}'.format(avg_psnr))
