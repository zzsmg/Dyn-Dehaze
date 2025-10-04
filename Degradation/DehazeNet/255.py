import cv2
import numpy as np
import os

pth1 = '/data/Pytorch_Porjects/DWT-FFC/datasets/combined_dataset/Test_offi/prior_net/'
pth2 = '/data/Pytorch_Porjects/DWT-FFC/datasets/combined_dataset/Test_offi/prior/'
files = os.listdir(pth1)
for file in files:
    prior = cv2.imread(pth1 + file, cv2.IMREAD_GRAYSCALE)
    normalized_img = cv2.normalize(prior, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    prior_255 = (normalized_img * 255).astype(np.uint8)
    cv2.imwrite(pth2 + file, prior_255)

