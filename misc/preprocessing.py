from glob import glob
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os


def prepare_sem():
    simulation_sem = sorted(glob('/data2/samsung/simulation_data/SEM/*/*/*.png'))
    train_sem = sorted(glob('/data2/samsung/train/SEM/*/*/*.png'))
    test_sem = sorted(glob('/data2/samsung/test/SEM/*.png'))

    return simulation_sem, train_sem, test_sem

def prepare_depth():
    simulation_depth_1 = sorted(glob(f'/data2/samsung/simulation_data/Depth/Case_1/*/*.png'))
    simulation_depth_2 = sorted(glob(f'/data2/samsung/simulation_data/Depth/Case_2/*/*.png'))
    simulation_depth_3 = sorted(glob(f'/data2/samsung/simulation_data/Depth/Case_3/*/*.png'))
    simulation_depth_4 = sorted(glob(f'/data2/samsung/simulation_data/Depth/Case_4/*/*.png'))
    return simulation_depth_1, simulation_depth_2, simulation_depth_3, simulation_depth_4

def load_imgs(img_paths):
    imgs = []
    save_img_paths = []
    save_folder_paths = []

    for img_path in img_paths:
        splited_path = img_path.split('/')
        splited_path = splited_path[:3] + ['preprocessed_data'] + splited_path[3:]
        save_img_path = "/".join(splited_path)
        save_folder_path = "/".join(splited_path[:-1])
        save_img_paths.append(save_img_path)
        save_folder_paths.append(save_folder_path)

    for save_folder_path in set(save_folder_paths):
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)

    for img_path in tqdm(img_paths):
        img = cv2.imread(img_path, 0)
        imgs.append(img)

    imgs = np.array(imgs)

    return [imgs, save_img_paths]

def preprocess(img_paths, new_max=0.9, new_min=-0.9):
    imgs, save_paths = load_imgs(img_paths)

    concat_imgs = imgs.reshape(-1, imgs.shape[-1])
    #MinMaxNormalization
    max, min = concat_imgs.max(), concat_imgs.min()

    normalized_imgs = (imgs - min) * (new_max - new_min) / (max - min) + new_min

    for normalized_img, save_path in zip(normalized_imgs, save_paths):
        cv2.imwrite(save_path, normalized_img, cv2.IMREAD_GRAYSCALE)

if __name__ == '__main__':
    simulation_sem, train_sem, test_sem = prepare_sem()
    simulation_depth_1, simulation_depth_2, simulation_depth_3, simulation_depth_4, = prepare_depth()

    preprocess(simulation_sem)
    preprocess(train_sem)
    preprocess(test_sem)

    preprocess(simulation_depth_1)
    preprocess(simulation_depth_2)
    preprocess(simulation_depth_3)
    preprocess(simulation_depth_4)

