from scipy.ndimage.morphology import (distance_transform_cdt,
                                      distance_transform_edt)
from tensorpack.dataflow.imgaug import ImageAugmentor
from tensorpack.utils.utils import get_rng
from scipy.ndimage import measurements
import numpy as np
import cv2

class GenInstance(ImageAugmentor):
    def __init__(self, crop_shape=None):
        super(GenInstance, self).__init__()
        self.crop_shape = crop_shape

    def reset_state(self):
        self.rng = get_rng(self)

    def cropping_center(self, x, crop_shape, batch=False):
        orig_shape = x.shape
        if not batch:
            h0 = int((orig_shape[0] - crop_shape[0]) * 0.5)
            w0 = int((orig_shape[1] - crop_shape[1]) * 0.5)
            x = x[h0:h0 + crop_shape[0], w0:w0 + crop_shape[1]]
        else:
            h0 = int((orig_shape[1] - crop_shape[0]) * 0.5)
            w0 = int((orig_shape[2] - crop_shape[1]) * 0.5)
            x = x[:, h0:h0 + crop_shape[0], w0:w0 + crop_shape[1]]
        return x

    def _fix_mirror_padding(self, ann):
        """
        Deal with duplicated instances due to mirroring in interpolation
        during shape augmentation (scale, rotation etc.)
        """
        current_max_id = np.amax(ann)
        inst_list = list(np.unique(ann))
        try:
            inst_list.remove(0)  # 0 is background
        except:
            inst_list = inst_list
        for inst_id in inst_list:
            inst_map = np.array(ann == inst_id, np.uint8)
            remapped_ids = measurements.label(inst_map)[0]
            remapped_ids[remapped_ids > 1] += int(current_max_id)
            ann[remapped_ids > 1] = remapped_ids[remapped_ids > 1]
            current_max_id = np.amax(ann)
        return ann

####


####
class GenInstanceOrd(GenInstance):
    """
        Generate an ordinal distance map based on the instance map
        First, the euclidead distance map will be calculated. Then, the ordinal map is generated based on the euclidean distance map
    """
    def _augment(self, img):
        def gen_ord(self, euc_map):
            lut_gt = [1, 0.83, 0.68, 0.54, 0.41, 0.29, 0.19, 0.09, 0]
            # lut_gt = [1.0, 0.68, 0.41, 0.19, 0.0]
            # lut_gt = [1.0, 0.31, 0.09, 0.02, 0.0]
            # lut_gt = [1.0, 0.45, 0.18, 0.06, 0.0]
            # lut_gt = [1.0, 0.51, 0.24, 0.09, 0.0]
            # lut_gt = [1.0, 0.59, 0.33, 0.17, 0.06, 0.0]
            zeros = np.zeros_like(euc_map)
            ones = np.ones_like(euc_map)
            decoded_label = np.full(euc_map.shape, 0, dtype=np.float32)
            for k in range(len(lut_gt) - 1):
                if k != len(lut_gt) - 2:
                    decoded_label += np.where((euc_map <= lut_gt[k]) & (euc_map > lut_gt[k + 1]), ones * (k + 1), zeros)
                else:
                    decoded_label += np.where((euc_map <= lut_gt[k]) & (euc_map >= lut_gt[k + 1]), ones * (k + 1),
                                              zeros)
            return decoded_label
        orig_ann = img[..., 0].astype(np.int32)  # instance ID map
        orig_ann[orig_ann == orig_ann[0, 0]] = 0
        orig_ann[orig_ann != 0] = 255

        contours = cv2.findContours(orig_ann.astype('uint8'), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)[0]
        largest_contour = sorted(contours, key=cv2.contourArea)[-1]
        remove_ann = cv2.drawContours(np.zeros_like(orig_ann, dtype=np.float32), [largest_contour], -1, 255, -1)

        ##1
        inst_id_map = np.copy(np.array(remove_ann == 255, dtype=np.uint8))
        M = cv2.moments(inst_id_map)
        cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
        inst_id_map[remove_ann != 255] = 2
        inst_id_map[int(cy), int(cx)] = 0
        inst_id_map = distance_transform_edt(inst_id_map)
        inst_id_map[remove_ann != 255] = 0
        inst_id_map = inst_id_map / np.max(inst_id_map)

        ##2
        # inst_id_map = cv2.distanceTransform(remove_ann.astype('uint8'), cv2.DIST_L2, 5)
        # inst_id_map_1[remove_ann != 255] = 0
        # inst_id_map = 1 - (inst_id_map / np.max(inst_id_map))

        ##3
        # contour = cv2.findContours(remove_ann.astype('uint8'), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)[0][0]
        # ellipse = cv2.ellipse(np.zeros_like(remove_ann, dtype=np.float32), cv2.fitEllipse(contour), 1, -1)
        # inst_id_map = cv2.distanceTransform(ellipse.astype('uint8'), cv2.DIST_L2, 5)
        # inst_id_map_3[remove_ann != 255] = 0
        # inst_id_map = 1 - (inst_id_map / np.max(inst_id_map))

        mask = np.zeros_like(remove_ann, dtype=np.float32)
        mask[remove_ann == 255] = inst_id_map[remove_ann == 255]
        ord_map = gen_ord(mask)


        img = img.astype('float32')
        imgs = np.dstack([img, ord_map])
        return imgs

def Ord_DDD(img, random_state, parents, hooks):
    def gen_ord(euc_map):
        lut_gt = [1, 0.83, 0.68, 0.54, 0.41, 0.29, 0.19, 0.09, 0]
        zeros = np.zeros_like(euc_map)
        ones = np.ones_like(euc_map)
        decoded_label = np.full(euc_map.shape, 0, dtype=np.float32)
        for k in range(len(lut_gt) - 1):
            if k != len(lut_gt) - 2:
                decoded_label += np.where((euc_map <= lut_gt[k]) & (euc_map > lut_gt[k + 1]), ones * (k + 1), zeros)
            else:
                decoded_label += np.where((euc_map <= lut_gt[k]) & (euc_map >= lut_gt[k + 1]), ones * (k + 1),
                                          zeros)
        return decoded_label

    img = np.array(img)[0]
    orig_ann = img[..., 0]  # instance ID map
    orig_ann[orig_ann == orig_ann[0, 0]] = 0
    orig_ann[orig_ann != 0] = 255

    contours = cv2.findContours(orig_ann.astype('uint8'), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)[0]
    largest_contour = sorted(contours, key=cv2.contourArea)[-1]
    remove_ann = cv2.drawContours(np.zeros_like(orig_ann, dtype=np.float32), [largest_contour], -1, 255, -1)

    ##1
    inst_id_map = np.copy(np.array(remove_ann == 255, dtype=np.uint8))
    M = cv2.moments(inst_id_map)
    cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
    inst_id_map[remove_ann != 255] = 2
    inst_id_map[int(cy), int(cx)] = 0
    inst_id_map = distance_transform_edt(inst_id_map)
    inst_id_map[remove_ann != 255] = 0
    inst_id_map = inst_id_map / np.max(inst_id_map)

    ##2
    # inst_id_map = cv2.distanceTransform(remove_ann.astype('uint8'), cv2.DIST_L2, 5)
    # inst_id_map_1[remove_ann != 255] = 0
    # inst_id_map = 1 - (inst_id_map / np.max(inst_id_map))

    ##3
    # contour = cv2.findContours(remove_ann.astype('uint8'), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)[0][0]
    # ellipse = cv2.ellipse(np.zeros_like(remove_ann, dtype=np.float32), cv2.fitEllipse(contour), 1, -1)
    # inst_id_map = cv2.distanceTransform(ellipse.astype('uint8'), cv2.DIST_L2, 5)
    # inst_id_map_3[remove_ann != 255] = 0
    # inst_id_map = 1 - (inst_id_map / np.max(inst_id_map))

    mask = np.zeros_like(remove_ann, dtype=np.float32)
    mask[remove_ann == 255] = inst_id_map[remove_ann == 255]
    ord_map = gen_ord(mask)
    ord_map = np.expand_dims(ord_map, axis=-1)
    return [ord_map]

def Ord_SDD(img, random_state, parents, hooks):
    def gen_ord(euc_map):
        lut_gt = [0.9, 0.435, 0.243, 0.095, -0.03, -0.139, -0.238, -0.33, -0.415, -0.494, -0.57, -0.641, -0.71, -0.776, -0.839, -0.9]
        zeros = np.zeros_like(euc_map)
        ones = np.ones_like(euc_map)
        decoded_label = np.full(euc_map.shape, 0, dtype=np.float32)
        for k in range(len(lut_gt) - 1):
            if k != len(lut_gt) - 2:
                decoded_label += np.where((euc_map <= lut_gt[k]) & (euc_map > lut_gt[k + 1]), ones * (k + 1), zeros)
            else:
                decoded_label += np.where((euc_map <= lut_gt[k]) & (euc_map >= lut_gt[k + 1]), ones * (k + 1),
                                          zeros)
        return decoded_label

    img = np.array(img)[0]
    ord_map = gen_ord(img)

    return [ord_map]

def Ord_Uniform(img, random_state, parents, hooks):
    def gen_ord(euc_map):
        lut_gt = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170]
        zeros = np.zeros_like(euc_map)
        ones = np.ones_like(euc_map)
        decoded_label = np.full(euc_map.shape, 0, dtype=np.float32)
        for k in range(len(lut_gt) - 1):
            decoded_label += np.where((euc_map > lut_gt[k]) & (euc_map <= lut_gt[k + 1]), ones * (k + 1), zeros)
        return decoded_label

    img = np.array(img)[0]
    ord_map = gen_ord(img)

    return [ord_map]

class GaussianBlur(ImageAugmentor):
    """ Gaussian blur the image with random window size"""
    def __init__(self, max_size=3):
        """
        Args:
            max_size (int): max possible Gaussian window size would be 2 * max_size + 1
        """
        super(GaussianBlur, self).__init__()
        self.max_size = max_size

    def _get_augment_params(self, img):
        sx, sy = self.rng.randint(1, self.max_size, size=(2,))
        sx = sx * 2 + 1
        sy = sy * 2 + 1
        return sx, sy

    def _augment(self, img, s):
        return np.reshape(cv2.GaussianBlur(img, s, sigmaX=0, sigmaY=0,
                                           borderType=cv2.BORDER_REPLICATE), img.shape)

####
class BinarizeLabel(ImageAugmentor):
    """ Convert labels to binary maps"""
    def __init__(self):
        super(BinarizeLabel, self).__init__()

    def _get_augment_params(self, img):
        return None

    def _augment(self, img, s):
        img = np.copy(img)
        arr = img[...,0]
        arr[arr > 0] = 1
        return img

####
class MedianBlur(ImageAugmentor):
    """ Median blur the image with random window size"""
    def __init__(self, max_size=3):
        """
        Args:
            max_size (int): max possible window size
                            would be 2 * max_size + 1
        """
        super(MedianBlur, self).__init__()
        self.max_size = max_size

    def _get_augment_params(self, img):
        s = self.rng.randint(1, self.max_size)
        s = s * 2 + 1
        return s

    def _augment(self, img, ksize):
        return cv2.medianBlur(img, ksize)