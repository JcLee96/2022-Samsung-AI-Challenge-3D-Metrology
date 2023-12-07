from glob import glob
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import math

def prepare_test_sem():
    test_sem_paths = sorted(glob('/data2/samsung/test/SEM/*.png'))
    return test_sem_paths

def prepare_result_sem():
    result_sem_paths = sorted(glob('/path/to/project_samsung2/result/submission/*.png'))
    return result_sem_paths

def prepare_train_sem():
    train_sem_paths = sorted(glob('/data2/samsung/train/SEM/*/*/*.png'))
    return train_sem_paths

def prepare_train_sem_per_Depth():
    train_sem_paths_1 = sorted(glob(f'/data2/samsung/train/SEM/Depth_110/*/*.png'))
    train_sem_paths_2 = sorted(glob(f'/data2/samsung/train/SEM/Depth_120/*/*.png'))
    train_sem_paths_3 = sorted(glob(f'/data2/samsung/train/SEM/Depth_130/*/*.png'))
    train_sem_paths_4 = sorted(glob(f'/data2/samsung/train/SEM/Depth_140/*/*.png'))
    return train_sem_paths_1, train_sem_paths_2, train_sem_paths_3, train_sem_paths_4

def prepare_simulation_sem():
    simulation_sem_paths = sorted(glob('/data2/samsung/simulation_data/SEM/*/*/*.png'))
    return simulation_sem_paths

def prepare_simulation_sem_per_Case():
    simulation_sem_paths_1 = sorted(glob(f'/data2/samsung/simulation_data/SEM/Case_1/*/*.png'))
    simulation_sem_paths_2 = sorted(glob(f'/data2/samsung/simulation_data/SEM/Case_2/*/*.png'))
    simulation_sem_paths_3 = sorted(glob(f'/data2/samsung/simulation_data/SEM/Case_3/*/*.png'))
    simulation_sem_paths_4 = sorted(glob(f'/data2/samsung/simulation_data/SEM/Case_4/*/*.png'))
    return simulation_sem_paths_1, simulation_sem_paths_2, simulation_sem_paths_3, simulation_sem_paths_4

def prepare_simulation_sem_per_8x(Case):
    simulation_sem_paths_1 = sorted(glob(f'/data2/samsung/simulation_data/SEM/Case_{Case}/80/*.png'))
    simulation_sem_paths_2 = sorted(glob(f'/data2/samsung/simulation_data/SEM/Case_{Case}/81/*.png'))
    simulation_sem_paths_3 = sorted(glob(f'/data2/samsung/simulation_data/SEM/Case_{Case}/82/*.png'))
    simulation_sem_paths_4 = sorted(glob(f'/data2/samsung/simulation_data/SEM/Case_{Case}/83/*.png'))
    simulation_sem_paths_5 = sorted(glob(f'/data2/samsung/simulation_data/SEM/Case_{Case}/84/*.png'))
    return simulation_sem_paths_1, simulation_sem_paths_2, simulation_sem_paths_3, simulation_sem_paths_4, simulation_sem_paths_5

def prepare_simulation_depth():
    simulation_depth_paths = sorted(glob('/data2/samsung/simulation_data/Depth/*/*/*.png'))
    return simulation_depth_paths

def prepare_simulation_depth_per_Case():
    simulation_depth_paths_1 = sorted(glob(f'/data2/samsung/simulation_data/Depth/Case_1/*/*.png'))
    simulation_depth_paths_2 = sorted(glob(f'/data2/samsung/simulation_data/Depth/Case_2/*/*.png'))
    simulation_depth_paths_3 = sorted(glob(f'/data2/samsung/simulation_data/Depth/Case_3/*/*.png'))
    simulation_depth_paths_4 = sorted(glob(f'/data2/samsung/simulation_data/Depth/Case_4/*/*.png'))
    return simulation_depth_paths_1, simulation_depth_paths_2, simulation_depth_paths_3, simulation_depth_paths_4

def prepare_simulation_depth_per_8x(Case):
    simulation_depth_paths_1 = sorted(glob(f'/data2/samsung/simulation_data/Depth/Case_{Case}/80/*.png'))
    simulation_depth_paths_2 = sorted(glob(f'/data2/samsung/simulation_data/Depth/Case_{Case}/81/*.png'))
    simulation_depth_paths_3 = sorted(glob(f'/data2/samsung/simulation_data/Depth/Case_{Case}/82/*.png'))
    simulation_depth_paths_4 = sorted(glob(f'/data2/samsung/simulation_data/Depth/Case_{Case}/83/*.png'))
    simulation_depth_paths_5 = sorted(glob(f'/data2/samsung/simulation_data/Depth/Case_{Case}/84/*.png'))
    return simulation_depth_paths_1, simulation_depth_paths_2, simulation_depth_paths_3, simulation_depth_paths_4, simulation_depth_paths_5

def prepare_sem():
    simulation_sem_paths = sorted(glob('/data2/samsung/simulation_data/SEM/*/*/*.png'))

    train_sem_paths = sorted(glob('/data2/samsung/train/SEM/*/*/*.png'))
    test_sem_paths = sorted(glob('/data2/samsung/test/SEM/*.png'))

    return simulation_sem_paths, train_sem_paths, test_sem_paths

def load_imgs(img_paths):
    imgs = []

    for idx, img_path in tqdm(enumerate(img_paths)):
        img = cv2.imread(img_path, 0)
        imgs.append(img)

    imgs = np.array(imgs)

    return imgs

def calcHist(imgs):
    imgs = imgs.reshape(-1, imgs.shape[-1])

    hist = cv2.calcHist([imgs], [0], None, [256], [0, 256])
    hist = hist / imgs.size * 100
    info = dict(max=round(imgs.max(), 3),
                min=round(imgs.min(), 3),
                avg=round(imgs.mean(), 3),
                std=round(imgs.std(), 3))

    return hist, info
def drawHist_per_dataset(paths, prefix, mode='depth', imgs_min=0, imgs_max=0):
    imgs = load_imgs(paths)

    imgs_avg = np.round(imgs.mean(0),0).astype('uint8')
    imgs_avg_max_pixel = imgs_avg.max()
    imgs_avg_min_pixel = imgs_avg.min()

    #normalization
    #imgs = (imgs-imgs_min).astype(float) * 255 / (imgs_max-imgs_min)

    #calcHistgram
    hist_sem, hist_info = calcHist(imgs_avg)

    # #Clamp
    # if mode == 'depth':
    #     for i in range(256):
    #         if hist_sem[i] > 1.0:
    #             hist_sem[i] = 1.0
    #     max_value = 1.0
    # else:
    #     max_value = hist_sem.max()

    # #SDD or DDD
    # list = [255, 174, 141, 115, 94, 75, 57, 42, 27, 13, 0]
    # max_values = [max_value if idx in list else 0 for idx in range(256)]

    # plt.imshow(imgs_avg, cmap='gray', vmin=0, vmax=255)
    # plt.title(prefix)

    fig = plt.figure(figsize=(15, 3))
    spec = gridspec.GridSpec(ncols=3, nrows=1,
                             width_ratios=[12, 2, 1])

    ax0 = fig.add_subplot(spec[0])
    ax0.bar(np.arange(256), hist_sem.squeeze(), color='b')
    ax0.set_xlim([0, 256])
    ax0.set_xticks([i*10 for i in range(26)])
    ax0.set_title(f'{prefix} Dataset Histogram')

    ax1 = fig.add_subplot(spec[1])
    ax1.imshow(imgs_avg, cmap='gray', vmin=0, vmax=255)
    ax1.set_title(prefix)
    ax1.axis('off')

    ax2 = fig.add_subplot(spec[2])
    ax2.text(0, 0.8, f"Max : {hist_info['max']}", fontsize=20)
    ax2.text(0, 0.6, f"Min : {hist_info['min']}", fontsize=20)
    ax2.text(0, 0.4, f"Avg : {hist_info['avg']}", fontsize=20)
    ax2.text(0, 0.2, f"Std : {hist_info['std']}", fontsize=20)
    ax2.axis('off')


    plt.savefig(f'/path/to/project_samsung2/histogram/ImgSUM_{prefix}.png', dpi=300, pad_inches=0)

def Ord_Uniform(img):

    def gen_ord(euc_map):
        lut_gt = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170]
        zeros = np.zeros_like(euc_map)
        ones = np.ones_like(euc_map)
        decoded_label = np.full(euc_map.shape, 0, dtype=np.float32)
        for k in range(len(lut_gt) - 1):
            decoded_label += np.where((euc_map > lut_gt[k]) & (euc_map <= lut_gt[k + 1]), ones * (k + 1), zeros)
        return decoded_label

    img = np.array(img)
    ord_map = gen_ord(img)
    return ord_map

def oridnal_label_augs(paths):
    imgs = load_imgs(paths)
    ord_maps = []
    for img in imgs:
       ord_map = Ord_Uniform(img)
       ord_maps.append(ord_map)
    x=0


if __name__ == '__main__':
    #simulation_sem = prepare_simulation_sem()
    # simulation_depth = prepare_simulation_depth()
    # oridnal_label_augs(simulation_depth)
    simulation_sem_1, simulation_sem_2, simulation_sem_3, simulation_sem_4 = prepare_simulation_sem_per_Case()
    simulation_depth_1, simulation_depth_2, simulation_depth_3, simulation_depth_4 = prepare_simulation_depth_per_Case()
    oridnal_label_augs(simulation_depth_4)
    # drawHist_per_dataset(simulation_sem_1, 'simulation_sem_case 1', 'sem', 0, 212)
    # drawHist_per_dataset(simulation_depth_1, 'simulation_depth_case 1', 'depth', 0, 170)
    # drawHist_per_dataset(simulation_sem_2, 'simulation_sem_case 2', 'sem', 0, 212)
    # drawHist_per_dataset(simulation_depth_2, 'simulation_depth_case 2', 'depth', 0, 170)
    # drawHist_per_dataset(simulation_sem_3, 'simulation_sem_case 3', 'sem', 0, 212)
    # drawHist_per_dataset(simulation_depth_3, 'simulation_depth_case 3', 'depth', 0, 170)
    # drawHist_per_dataset(simulation_sem_4, 'simulation_sem_case 4', 'sem', 0, 212)
    # drawHist_per_dataset(simulation_depth_4, 'simulation_depth_case 4', 'depth', 0, 170)
    #
    # train_sem = prepare_train_sem()
    # drawHist_per_dataset(train_sem, 'train_sem', 'sem', 0, 212)
    #
    # train_sem_1, train_sem_2, train_sem_3, train_sem_4 = prepare_train_sem_per_Depth()
    # drawHist_per_dataset(train_sem_1, 'train_sem_depth 110', 'sem', 0, 212)
    # drawHist_per_dataset(train_sem_2, 'train_sem_depth 120', 'sem', 0, 212)
    # drawHist_per_dataset(train_sem_3, 'train_sem_depth 130', 'sem', 0, 212)
    # drawHist_per_dataset(train_sem_4, 'train_sem_depth 140', 'sem', 0, 212)
    #
    # test_sem = prepare_test_sem()
    # drawHist_per_dataset(test_sem, 'test_sem', 'sem', 0, 212)
    # #
    # # result_sem = prepare_result_sem()
    # # drawHist_per_dataset(result_sem, 'result_sem', 'depth', 0, 170)
    # #
