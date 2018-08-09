#!/usr/bin/env python3

import os
from pymoth import Sequence
from pymoth import Namespace


class DataSet(Namespace):

    def __init__(self, directory):
        Namespace.__init__(self)
        # For each sub directory (typically 'train' and 'test')
        for sub_dir in os.listdir(directory):
            # Add a new Namespace under the name of this sub directory
            self.add({sub_dir: Namespace()})
            # For each data set in the sub directory
            for data_set in os.listdir("%s/%s" % (directory, sub_dir)):
                # Get the expected paths to det.tex, gt.tex, img1 and seqinfo.ini
                det_path = "%s/%s/%s/det/det.txt" % (directory, sub_dir, data_set)
                gt_path = "%s/%s/%s/gt/gt.txt" % (directory, sub_dir, data_set)
                img_dir = "%s/%s/%s/img1" % (directory, sub_dir, data_set)
                seq_path = "%s/%s/%s/seqinfo.ini" % (directory, sub_dir, data_set)
                # Ensure the data set name can be used a variable name by replacing hyphens with underscores
                data_set = data_set.replace("-", "_")
                # Add a new Namespace under the name of this data set
                self.add({data_set: Namespace()}, sub_space=sub_dir)
                # If det.txt is found, name a new Namespace under det and load the data for each frame
                if os.path.isfile(det_path):
                    self.add({"det": Sequence()}, sub_space=[sub_dir, data_set])
                    self.get(key=[sub_dir, data_set, "det"]).load_frames(img_dir, det_path, seq_path)
                # If det.txt is found, name a new Namespace under gt and load the data for each frame
                if os.path.isfile(gt_path):
                    self.add({"gt": Sequence()}, sub_space=[sub_dir, data_set])
                    self.get(key=[sub_dir, data_set, "gt"]).load_frames(img_dir, gt_path, seq_path)
