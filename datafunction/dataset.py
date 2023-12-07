import cv2
import torch.utils.data as data
import numpy as np
from glob import glob

def prepare_sem_data():
    simulation_sem_paths = sorted(glob('/data2/lee/samsung/simulation_data/SEM/*/*/*.png'))
    simulation_depth_paths = sorted(glob('/data2/lee/samsung/simulation_data/Depth/*/*/*.png') + glob('/data2/lee/samsung/simulation_data/Depth/*/*/*.png'))
    test_paths = sorted(glob('/data2/lee/samsung/test/SEM/*.png'))

    data_len = len(simulation_sem_paths)  # == len(simulation_depth_paths)

    train_sem_paths = simulation_sem_paths[:int(data_len * 0.8)]
    train_depth_paths = simulation_depth_paths[:int(data_len * 0.8)]

    val_sem_paths = simulation_sem_paths[int(data_len * 0.8):]
    val_depth_paths = simulation_depth_paths[int(data_len * 0.8):]

    test_sem_paths = test_paths
    test_depth_paths = ['' for i in test_paths]

    train_set = list(zip(train_sem_paths, train_depth_paths))
    valid_set = list(zip(val_sem_paths, val_depth_paths))
    test_set = list(zip(test_sem_paths, test_depth_paths))

    return train_set, valid_set, test_set

def MinMaxNormalization(img, img_min, img_max):
    # min, max = img.min(), img.max()
    # normalized_img = (img - min) * (new_max - new_min) / (max - min) + new_min

    normalized_img = (img - img_min) / (img_max - img_min)

    return normalized_img, [img_min, img_max]

def Normalization(img, new_min, new_max):
    min, max = 0, 255
    normalized_img = (img - min) * (new_max - new_min) / (max - min) + new_min

    return normalized_img
class DatasetSerialSEM(data.Dataset):

    def __init__(self, pair_list, shape_augs=None, input_augs=None, label_augs=None, mode='train'):
        self.pair_list = pair_list
        self.shape_augs = shape_augs
        self.input_augs = input_augs
        self.label_augs = label_augs
        self.mode = mode

    def __getitem__(self, idx):
        pair = self.pair_list[idx]
        img_path = pair[0]
        input_sem = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        input_sem = np.expand_dims(input_sem, axis=-1)#.transpose(2, 0, 1)  # H, W, C

        if self.mode == 'test':
            if self.shape_augs is not None:
                shape_augs = self.shape_augs.to_deterministic()
                input_sem = shape_augs.augment_image(input_sem)

            if self.input_augs is not None:
                input_sem = self.input_augs.augment_image(input_sem)

            input_sem, minmax = MinMaxNormalization(input_sem, 0, 212)

            # input_sem = Normalization(input_sem, new_min=0, new_max=1)
            return input_sem, img_path, minmax

        elif self.mode == 'train':
            input_depth = cv2.imread(pair[1], cv2.IMREAD_GRAYSCALE)
            input_depth = np.expand_dims(input_depth, axis=-1)#.transpose(2, 0, 1)
            if self.shape_augs is not None:
                shape_augs = self.shape_augs.to_deterministic()
                input_sem = shape_augs.augment_image(input_sem)
                input_depth = shape_augs.augment_image(input_depth)

            if self.input_augs is not None:
                input_sem = self.input_augs.augment_image(input_sem)

            if self.label_augs is not None:
                input_ord = input_depth.copy()
                input_ord = self.label_augs.augment_image(input_ord)

            input_sem, _ = MinMaxNormalization(input_sem, 0, 212)
            input_depth, _ = MinMaxNormalization(input_depth, 0, 170)

            # input_sem = Normalization(input_sem, new_min=0, new_max=1)
            # input_depth = Normalization(input_depth, new_min=0, new_max=1)


            return input_sem, input_depth, input_ord

        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0],
        #                          std=[1])
        # ])
        # input_sem = np.array(transform(input_sem)).transpose(1, 2, 0)
        # input_depth = np.array(transform(input_depth)).transpose(1, 2, 0)

    def __len__(self):
        return len(self.pair_list)