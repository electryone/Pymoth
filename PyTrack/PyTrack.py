#!/usr/bin/env python3

import os

from PyTrack import Sequence
from PyTrack import Namespace
from PyTrack.utils import chunks


class PyTrack(Namespace):

    def __init__(self, directory):
        Namespace.__init__(self)
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
                    self.get()[sub_dir].get()[data_set_].get()["det"] = Sequence()
                    self.get()[sub_dir].get()[data_set_].get()["det"].load_frames(img_dir, det_path, seq_path)
                if os.path.isfile(gt_path):
                    self.get()[sub_dir].get()[data_set_].get()["gt"] = Sequence()
                    self.get()[sub_dir].get()[data_set_].get()["gt"].load_frames(img_dir, gt_path, seq_path)

    @staticmethod
    def split(tracklets, size):
        """
        :param tracklets:
        :param size:
        :return: list: of equal length uninterrupted sequences of instances
        """
        sequences = []
        for id in tracklets:
            for sequence in id:
                sequences.append(sequence)
        euqi_sequences = []
        for sequence in sequences:
            for chunk in chunks(sequence, size):
                if len(chunk) == size:
                    euqi_sequences.append(chunk)
        return euqi_sequences
