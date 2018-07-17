#!/usr/bin/env python3

import cv2
import numpy as np
import matplotlib.pyplot as plt

from . import Sequence
from .utils import load_info


class PyTrack(object):

    def __init__(self, img_dirs, det_paths, info_paths, gt_paths=None):
        self.info = {}
        self.det = {}
        self.gt = {}
        self.__load_sequences(img_dirs, det_paths, info_paths, gt_paths)

    def __load_sequences(self, img_dirs, det_paths, info_paths, gt_paths):
        """
        :return:
        """
        info_list = [load_info(path) for path in info_paths]
        for img, det, info, gt in zip(img_dirs, det_paths, info_list, gt_paths):
            self.det[info.name] = Sequence()
            self.det[info.name].load_frames(img, det, info)
            self.info[info.name] = info
            if gt is not None:
                self.gt[info.name] = Sequence()
                self.gt[info.name].load_frames(img, gt, info)
