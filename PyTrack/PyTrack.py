#!/usr/bin/env python3

import os

from PyTrack import Sequence
from PyTrack import Namespace


class PyTrack(Namespace):

    def __init__(self, directory):
        Namespace.__init__(self)
        self.__load_dataset(directory)

    def __load_dataset(self, directory):
        for sub_dir in os.listdir(directory):
            vars(self)[sub_dir] = Namespace()
            for data_set in os.listdir("%s/%s" % (directory, sub_dir)):
                data_set_ = data_set.replace("-", "_")
                vars(vars(self)[sub_dir])[data_set_] = Namespace()
                det_path = "%s/%s/%s/det/det.txt" % (directory, sub_dir, data_set)
                gt_path = "%s/%s/%s/gt/gt.txt" % (directory, sub_dir, data_set)
                img_dir = "%s/%s/%s/img1" % (directory, sub_dir, data_set)
                seq_path = "%s/%s/%s/seqinfo.ini" % (directory, sub_dir, data_set)
                if os.path.isfile(det_path):
                    vars(vars(vars(self)[sub_dir])[data_set_])["det"] = Sequence()
                    vars(vars(vars(self)[sub_dir])[data_set_])["det"].load_frames(img_dir, det_path, seq_path)
                if os.path.isfile(gt_path):
                    vars(vars(vars(self)[sub_dir])[data_set_])["gt"] = Sequence()
                    vars(vars(vars(self)[sub_dir])[data_set_])["gt"].load_frames(img_dir, gt_path, seq_path)
