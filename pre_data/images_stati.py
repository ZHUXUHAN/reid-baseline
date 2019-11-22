import cv2
from tqdm import tqdm
import numpy as np
import os

TRAIN_ROOT = "/home/xiangan/data_reid/trainA/train_set"
TEST_QUERY_ROOT = "/home/xiangan/data_reid/testA/query_a"
TEST_GALLERY_ROOT = "/home/xiangan/data_reid/testA/gallery_a"

if __name__ == '__main__':
    root_list = [TRAIN_ROOT, TEST_QUERY_ROOT, TEST_GALLERY_ROOT]
    b_mean_list = []
    g_mean_list = []
    r_mean_list = []
    b_std_list = []
    g_std_list = []
    r_std_list = []
    for r in root_list:
        for p in os.listdir(r):
            try:
                image = cv2.imread(os.path.join(r, p)) / 255
                b_mean_list.append(image[:, :, 0].mean())
                g_mean_list.append(image[:, :, 1].mean())
                r_mean_list.append(image[:, :, 2].mean())
                b_std_list.append(image[:, :, 0].std())
                g_std_list.append(image[:, :, 1].std())
                r_std_list.append(image[:, :, 2].std())
            except:
                print('error')
        print('b_mean', np.mean(b_mean_list))
        print('g_mean', np.mean(g_mean_list))
        print('r_mean', np.mean(r_mean_list))
        print('b_std', np.mean(b_std_list))
        print('g_std', np.mean(g_std_list))
        print('r_std', np.mean(r_std_list))
