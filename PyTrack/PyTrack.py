#!/usr/bin/env python3

import cv2
import numpy as np
import matplotlib.pyplot as plt

from . import Sequence
from . import Clock
from .utils import box2rect
from .utils import iou
from .utils import load_info
from .utils import nms
from .utils import print_dict


class PyTrack(object):

    def __init__(self, img_dirs, det_paths, info_paths, gt_paths=None, scale=1, width=1):
        self.img_dirs = img_dirs
        self.det_paths = det_paths
        self.gt_paths = gt_paths
        self.scale = scale
        self.width = width
        self.info = [load_info(path) for path in info_paths]
        self.det = {}
        self.gt = {}

        self.load_sequences()

    def load_sequences(self):
        """
        :return:
        """
        for img, det, info, gt in zip(self.img_dirs, self.det_paths, self.info, self.gt_paths):
            self.det[info.name] = Sequence()
            self.det[info.name].load_frames(img, det, info)
            if gt is not None:
                self.gt[info.name] = Sequence()
                self.gt[info.name].load_frames(img, gt, info)

    def get_iou(self, name, frame):
        """
        :param name:
        :param frame:
        :return:
        """
        # Get detection and ground truth boxes
        det_boxes = self.det[name].get_bounding_boxes(frame)
        gt_boxes = self.gt[name].get_bounding_boxes(frame)

        # get detection and ground truth rects
        det_rects = box2rect(det_boxes)
        gt_rects = box2rect(gt_boxes)

        # Get the iou matrix
        n = self.gt[name].get_n_instances(frame)
        iou_matrix = iou(np.concatenate((gt_rects, det_rects)))
        return iou_matrix[:n, n:]

    def get_fp0(self, name, tau_iou=0.5):
        """
        :param name:
        :param tau_iou:
        :return:
        """
        fp0 = Sequence()
        fp0.init_frames(info=self.info, img_dir=self.img_dirs)
        for frame in range(self.det[name].get_n_frames()):

            # Get association vector that assigns detections to ground truth objects
            association, _ = self.get_association(frame, tau_iou=tau_iou)

            # For each ground truth and corresponding detection
            for gt_index, det_index in enumerate(association.astype(int)):
                # If the ground truth instance is covered by a detection
                if det_index != -1:
                    fp0.add_instance(frame, self.det[name].get_instances(frame)[det_index])
        return fp0

    def get_association(self, name, frame, tau_iou=0.5):
        """
        :param name:
        :param frame:
        :param tau_iou:
        :return:
        """
        # Get matrix of iou values
        iou_matrix = self.get_iou(name, frame)

        # Apply iou threshold
        iou_matrix[iou_matrix < tau_iou] = 0

        # Apply non-maximum suppression
        iou_matrix = nms(iou_matrix)

        # Get association vector (defines which det is assigned to which gt, -1 denotes no detection)
        n_instances = self.gt[name].get_n_instances(frame)
        association = np.empty(n_instances)
        for i in range(n_instances):
            if not np.any(iou_matrix[i, :]):
                association[i] = -1
            else:
                association[i] = np.argmax(iou_matrix[i, :])
        return association, iou_matrix

    def evaluate_detections(self, name):
        """
        :return:
        """
        # Initialise dictionary of detection coverage over time for each object
        detection_dict = {}
        for item in self.gt[name].get_ids():
            array = np.empty(self.gt[name].get_n_frames())
            array.fill(-1)
            detection_dict[item] = array

        for frame in range(self.det[name].get_n_frames()):
            # Get association vector (defines which det is assigned to which gt, -1 denotes no detection)
            association, iou_matrix = self.get_association(frame)

            for gt_index, (gt_id, det_index) in enumerate(zip(self.gt[name].get_ids(frame), association.astype(int))):
                # If there is no ground truth for the current detection
                if det_index == -1:
                    detection_dict[gt_id][frame] = 0
                else:
                    detection_dict[gt_id][frame] = iou_matrix[gt_index, det_index]

                # Get list of tru positive detections
            true_positives = []
            for i, instance in enumerate(self.det[name].get_instances(frame)):
                if i in association:
                    true_positives.append(instance)

        # Initialise result dictionary
        results = {"MT": 0,
                   "ML": 0,
                   "PT": 0,
                   "TP": 0,
                   "FN": 0}
        for id in detection_dict:
            # Store fn as 0
            f = np.where(detection_dict[id] == 0)[0]            # frames were false negative
            fn = np.empty_like(f)
            fn.fill(id)
            plt.scatter(f, fn, marker="_", c="red")
            n_fn = f.shape[0]

            # Store tp as the id number
            f = np.where(detection_dict[id] > 0)[0]             # frames where true positive
            tp = np.empty_like(f)
            tp.fill(id)
            plt.scatter(f, tp, marker="_", c="blue")
            n_tp = f.shape[0]

            # Store results in a dictionary
            results["TP"] += n_tp
            results["FN"] += n_fn
            if n_tp / (n_tp + n_fn) > 0.8:
                results["MT"] += 1
            elif n_tp / (n_tp + n_fn) < 0.2:
                results["ML"] += 1
            else:
                results["PT"] += 1
        results["Detections"] = self.det[name].get_n_instances()
        results["Instances"] = self.gt[name].get_n_instances()
        results["FP"] = results["Detections"] - results["TP"]
        results["TPR"] = results["TP"] / results["Instances"]
        results["FNR"] = results["FN"] / results["Instances"]
        print_dict(results)

        # Plot graph
        plt.ylim(0)
        plt.xlim(0)
        plt.grid()
        plt.show()
        return results
