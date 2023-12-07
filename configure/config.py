import imgaug  # https://github.com/aleju/imgaug
from imgaug import augmenters as iaa
import imgaug as ia
import os
from misc.augs import *

####
class CONFIGURE(object):
    def __init__(self, _args=None):
        self.seed = 5
        self.init_lr = 1.0e-3
        self.lr_steps = 20  # decrease at every n-th epoch
        self.gamma = 0.2
        self.train_batch_size = 512
        self.infer_batch_size = 512
        self.nr_epochs = 40

        # nr of processes for parallel processing input
        self.nr_procs_train = 16
        self.nr_procs_valid = 16

        self.logging = True  # True for debug run only
        if _args is not None:
            self.__dict__.update(_args.__dict__)
            self.seed = _args.seed
            self.lr = _args.lr
            self.nr_epochs = _args.nr_epochs
            self.ordinal_class = _args.ordinal_class
            self.gpu = _args.gpu
            # self.ord_mode = _args.ord_mode
            # self.loss = _args.loss
            print(_args.__dict__)

    def train_augmentors(self):
        sometimes = lambda aug: iaa.Sometimes(0.2, aug)
        # apply the following augmenters to most images
        shape_augs = iaa.Sequential([iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                                     iaa.Flipud(0.5),  # vertically flip 50% of all images
                                     # sometimes(iaa.Affine(rotate=(-45, 45),  # rotate by -45 to +45 degrees
                                     #                      shear=(-16, 16),  # shear by -16 to +16 degrees
                                     #                      order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                                     #                      cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                                     #                      mode='symmetric'  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                                     # ))
                                    ])

        input_augs = iaa.Sequential(
            [iaa.SomeOf((0, 5),
                        [iaa.OneOf([iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                                    iaa.AverageBlur(k=(2, 7)),
                                    # blur image using local means with kernel sizes between 2 and 7
                                    iaa.MedianBlur(k=(3, 11)),
                                    # blur image using local medians with kernel sizes between 2 and 7
                                    ]),
                         iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255)),
                         # add gaussian noise to images
                         iaa.Dropout((0.01, 0.1)),  # randomly remove up to 10% of the pixels
                         # change brightness of images (by -10 to 10 of original value)
                         iaa.LinearContrast((0.5, 2.0)),  # improve or worsen the contrast
                         #iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
                        ], random_order=True)
            ], random_order=True)

        ord_augmenter = iaa.Lambda(func_images=Ord_Uniform)
        label_augs = iaa.Sequential(ord_augmenter)
        return shape_augs, input_augs, label_augs

    ####
    def infer_augmentors(self):
        shape_augs = None
        input_augs = None
        ord_augmenter = iaa.Lambda(func_images=Ord_Uniform)
        label_augs = iaa.Sequential(ord_augmenter)
        return shape_augs, input_augs, label_augs

##########################################################################
