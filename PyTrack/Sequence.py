#!/usr/bin/env python3

"""
A Sequence object contains a list of frames, each frame has a list of instances (each with an id number).
"""

import os
import cv2
from itertools import count
import numpy as np
from keras.utils.generic_utils import Progbar
import time

from .Frame import Frame
from .Clock import Clock
from .utils import convert
from .utils import pairwise

# MOTChallenge variables and format
file_format = {"frame": 0,
               "id": 1,
               "bb_left": 2,
               "bb_top": 3,
               "bb_width": 4,
               "bb_height": 5,
               "conf": 6}


class Sequence(object):

    def __init__(self):
        self.info = None
        self.frames = []

    def load_frames(self, img_dir, label_paths, info):
        """
        Initialises sequence from files
        Use when all the sequence data is known beforehand
        Load the image path and label data into each frame
        :return: None
        """
        self.init_frames(info=info, img_dir=img_dir)
        with open(label_paths, "r") as file:
            for line in file:
                data = self.__extract_data(line)
                kwargs = {"id": data["id"],
                          "bounding_box": (data["bb_left"],
                                           data["bb_top"],
                                           data["bb_width"],
                                           data["bb_height"]),
                          "conf": data["conf"]}
                self.create_instance(data["frame"], kwargs)

    def init_frames(self, info=None, n=None, img_dir=None):
        if info is not None:
            self.info = info
            self.frames = [Frame(index=i) for i in range(self.info.seqLength)]
        elif n is not None:
            self.frames = [Frame(index=i) for i in range(n)]
        else:
            raise ValueError("an info Namespace or the number of frames must be given")
        if img_dir is not None:
            self.set_frame_paths(img_dir)

    def new_frame(self, img_path=None):
        self.frames.append(Frame(index=self.get_n_frames(), img_path=img_path))

    def set_frame_paths(self, img_dir):
        """
        :param img_dir:
        :return:
        """
        frame_paths = os.listdir(img_dir)
        for path in frame_paths:
            i = int(path.split(".")[0]) - 1
            self.set_frame_path(i, "%s/%s" % (img_dir, path))

    def set_frame_path(self, frame, path):
        """
        Set the image path for a frame
        :param frame: int: index to frame in self.frames
        :param path: str: path to frame image file
        :return: None
        """
        self.frames[frame].img_path = path
        for instance in self.frames[frame].instances:
            instance.path = path

    def create_instance(self, frame, kwargs):
        """
        :param frame:
        :param kwargs:
        :return:
        """
        self.frames[frame].create_instance(kwargs)

    def add_instance(self, frame, instance):
        """
        :param frame:
        :param instance:
        :return:
        """
        self.frames[frame].add_instance(instance)

    def get_images(self, width=1, scale=1, draw=False, show_ids=False):
        """
        :param width:
        :param scale:
        :param draw:
        :param show_ids:
        :return:
        """
        images = np.empty((self.get_n_frames(), self.info.imHeight, self.info.inWidth, 3),
                          dtype=np.uint8)
        for i, frame in enumerate(self.frames):
            images[i] = frame.get_image(width=width, scale=scale, draw=draw, show_ids=show_ids)
        return images

    def get_n_frames(self):
        """
        :return: int: the number of frames in the sequence
        """
        return len(self.frames)

    def get_n_ids(self):
        """
        :return: int: the number of target in the frame or sequence
        """
        return len(np.unique(self.get_ids()))

    def get_n_instances(self, id=None):
        """
        :param id: int: the id number of a target to be counted
        :return: int: the number of instances in the frame
        """
        if id is None:
            return len([1 for frame in self.frames for _ in frame.instances])
        else:
            return len([1 for frame in self.frames for instance in frame.instances if instance.id == id])

    def get_instances(self, id=None):
        """
        :return:
        """
        if id is None:
            return [instance for frame in self.frames for instance in frame.instances]
        else:
            return [instance for frame in self.frames for instance in frame.instances if instance.id == id]

    def get_ids(self):
        """
        :return:
        """
        return [instance.id for frame in self.frames for instance in frame.instances]

    def get_boxes(self, id=None):
        """
        :param id:
        :return:
        """
        boxes = np.empty((self.get_n_instances(id=id), 4))
        for i, instance in enumerate(self.get_instances(id=id)):
            boxes[i] = instance.bounding_box
        return boxes

    def get_rects(self, id=None):
        """
        :param id:
        :return:
        """
        rects = np.empty((self.get_n_instances(id=id), 4))
        for i, instance in enumerate(self.get_instances(id=id)):
            rects[i] = instance.rect
        return rects

    def get_xywh(self, id=None):
        """
        :param id:
        :return:
        """
        xywh = np.empty((self.get_n_instances(id=id), 4))
        for i, instance in enumerate(self.get_instances(id=id)):
            xywh[i] = instance.xywh
        return xywh

    def get_conf(self, id=None):
        """
        :param id:
        :return:
        """
        if id is None:
            return [instance.conf for frame in self.frames for instance in frame.instances]
        else:
            return [instance.conf for frame in self.frames for instance in frame.instances if instance.id == id]

    def get_appearances(self, id=None, shape=None):
        """
        :param id:
        :param shape:
        :return:
        """
        if shape is None:
            appearances = []
            if id is None:
                progress_bar = Progbar(self.get_n_frames(), width=30, verbose=1, interval=1)
                for i, frame in enumerate(self.frames):
                    appearances += frame.get_appearances(shape=shape)
                    progress_bar.update(i)
            else:
                progress_bar = Progbar(self.get_n_instances(id=id), width=30, verbose=1, interval=1)
                for i, instance in enumerate(self.get_instances(id=id)):
                    appearances.append(instance.get_appearance(shape=shape))
                    progress_bar.update(i)
        else:
            appearances = np.empty((self.get_n_instances(id=id), shape[0], shape[1], shape[2]),
                                   dtype=np.uint8)
            if id is None:
                progress_bar = Progbar(self.get_n_frames(), width=30, verbose=1, interval=1)
                j = 0
                for i, frame in enumerate(self.frames):
                    n = frame.get_n_instances()
                    appearances[j:j + n] = frame.get_appearances(shape=shape)
                    j += n
                    progress_bar.update(i)
            else:
                progress_bar = Progbar(self.get_n_instances(id=id), width=30, verbose=1, interval=1)
                for i, instance in enumerate(self.get_instances(id=id)):
                    appearances[i] = instance.get_appearance(shape=shape)
                    progress_bar.update(i)
        print("\n")
        return appearances

    def show(self, scale=1, width=1, draw=False, show_id=False):
        """
        :return: None
        """
        clock = Clock(self.info.frameRate)
        for frame in self.frames:
            image = frame.get_frame(width=width, scale=scale, draw=draw, show_id=show_id)
            cv2.imshow(self.info.name, image)
            cv2.waitKey(1)
            clock.toc()

    def get_appearance_pairs(self, shape=(128, 128, 3), seed=None):
        """
        :return:
        """
        if seed is not None:
            np.random.seed(seed)
        targets = self.get_appearances(shape=shape)
        n_targets = len(targets)
        x = []
        y = []
        print("Generating positive/negative pairs from %s targets" % n_targets)
        progress_bar = Progbar(n_targets, width=30, verbose=1, interval=1)
        for i, target in enumerate(targets):
            non_target = targets.copy()
            del non_target[i]
            non_target = np.concatenate(non_target, axis=0)
            n = target.shape[0] - 1
            x_t = np.empty(tuple([2 * n, 2] + list(shape)), dtype=np.uint8)
            y_t = np.empty((2 * n), dtype=np.uint8)
            for j, (target_0, target_1) in enumerate(pairwise(target)):
                x_t[j, 0] = target_0
                x_t[j, 1] = target_1
                y_t[j] = 1
                x_t[j + n, 0] = target_0
                x_t[j + n, 1] = non_target[np.random.randint(0, non_target.shape[0])]
                y_t[j + n] = 0
            x.append(x_t)
            y.append(y_t)
            progress_bar.update(i)
        print("\n")
        x = np.concatenate(x, axis=0)
        y = np.concatenate(y, axis=0)
        return x, y

    @staticmethod
    def __extract_data(line):
        """
        Extract label data from a line of the labels file
        :param line: str: line from a label file
        :return: dict: dictionary of data from file
        """
        line = line.replace("\n", "").split(",")
        data = {}
        for key, index in file_format.items():
            data[key] = convert(line[index])
        data["frame"] -= 1
        return data
