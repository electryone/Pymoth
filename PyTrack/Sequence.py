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

    def __init__(self, name="Sequence"):
        self.name = name
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
                self.create_instance(data["frame"],
                                     data["id"],
                                     bounding_box=(data["bb_left"],
                                                   data["bb_top"],
                                                   data["bb_width"],
                                                   data["bb_height"]),
                                     conf=data["conf"])

    def get_all_target_appearances(self, shape=(64, 64, 3)):
        n_frames = self.get_n_frames()
        n_targets = self.get_n_targets()
        n_instances = [self.get_n_instances(id=id+1) for id in range(n_targets)]
        appearances = [np.empty(tuple([n] + list(shape)), dtype=np.uint8) for n in n_instances]
        indexes = [count() for _ in range(n_targets)]
        print("Getting the appearance of each target from %s frames" % n_frames)
        progress_bar = Progbar(n_frames, width=30, verbose=1, interval=1)
        for i, frame in enumerate(self.frames):
            for appearance, id in zip(frame.get_appearances(shape=shape), frame.get_ids()):
                appearances[id - 1][next(indexes[id - 1])] = appearance
            progress_bar.update(i)
        print("\n")
        return appearances

    def get_appearance_pairs(self, shape=(64, 64, 3), seed=None):
        """
        :return:
        """
        if seed is not None:
            np.random.seed(seed)
        targets = self.get_all_target_appearances(shape=shape)
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

    def new_frame(self, img_path=None):
        self.frames.append(Frame(nb=len(self.frames), img_path=img_path))

    def init_frames(self, info=None, n=None, img_dir=None):
        if info is not None:
            self.info = info
            self.frames = [Frame(nb=i) for i in range(self.info.seqLength)]
        elif n is not None:
            self.frames = [Frame(nb=i) for i in range(n)]
        if img_dir is not None:
            self.set_frame_paths(img_dir)

    def create_instance(self, frame, id, bounding_box=None, coordinates=None, conf=None):
        """
        Create a new instance and append it to the appropriate frame
        :param frame: int: frame number
        :param id: int: instance identification number
        :param bounding_box: list: [bb_left, bb_top, bb_width, bb_height]
        :param coordinates: list: [x, y, z]
        :param conf: detection confidence (-1 if unknown)
        :return: None
        """
        self.frames[frame].create_instance(id,
                                           bounding_box=bounding_box,
                                           coordinates=coordinates,
                                           conf=conf)

    def add_instance(self, frame, instance):
        self.frames[frame].add_instance(instance)

    def get_scene(self, frame, width=1, scale=1, draw=False):
        """
        Display a particular frame
        :param frame: int: index to a frame in self.frames
        :param width: int: line width
        :param scale: int: scale factor
        :param draw: bool: whether or not to draw instances
        :return: np.array: image
        """
        return self.frames[frame].get_scene(width=width, scale=scale, draw=draw)

    def get_n_frames(self):
        """
        :return: int: the number of frames in the sequence
        """
        return len(self.frames)

    def get_instances(self, frame=None):
        """
        :param frame: int: index to a frame in self.frames
        :return: list: instances
        """
        if frame is None:
            instances = []
            for frame in self.frames:
                for instance in frame.get_instances():
                    instances.append(instance)
            return instances
        else:
            return self.frames[frame].get_instances()

    def get_n_instances(self, frame=None, id=None):
        """
        :param frame: int: the index to a frame in self.frames
        :param id: int: the id number of a target to be counted
        :return: int: the number of instances in the frame
        """
        if frame is not None and id is not None:
            raise ValueError("Can not return n_instances for a given frame and id")
        if id is not None:
            count = 0
            for frame in self.frames:
                for instance in frame.get_instances():
                    if instance.id == id:
                        count += 1
            return count
        elif frame is not None:
            return self.frames[frame].get_n_instances()
        else:
            return sum([f.get_n_instances() for f in self.frames])

    def get_target(self, id):
        """
        Get all instances of a given target
        :param id: int: unique id of target
        :return: list: list of instances
        """
        instances = []
        for frame in self.frames:
            for instance in frame.get_instances():
                if instance.id == id:
                    instances.append(instance)
        return instances

    def get_target_boxes(self, id):
        """
        Return boundinge boxes for a target for each frame the target appears in
        :param id: int: unique id of target
        :return: boxes: np.array: bounding boxes (left, top, w, h)
        """
        target = self.get_target(id)
        boxes = np.empty((len(target), 4))
        for i, instance in enumerate(target):
            boxes[i] = instance.get_box()
        return boxes

    def get_target_appearances(self, id, shape=None):
        """
        :param id: int, target unique id number
        :param shape: tuple: image height, width and number of channels
        :return: list or np.array: target appearance for each frame
        """
        target = self.get_target(id)
        if shape is None:
            appearances = []
            for instance in target:
                appearances.append(instance.get_appearance(shape=shape))
        else:
            appearances = np.empty((len(target), shape[0], shape[1], shape[2]), dtype=np.uint8)
            for i, instance in enumerate(target):
                appearances[i, :, :, :] = instance.get_appearance(shape=shape)
        return appearances

    def get_n_targets(self, frame=None):
        """
        :param frame: int: the frame of interest
        :return: int: the number of target in the frame or sequence
        """
        if frame is None:
            return len(np.unique(self.get_ids()))
        else:
            return len(self.get_ids(frame))

    def get_ids(self, frame=None):
        """
        :param frame: int: the index to a frame in self.frames
        :return: list: set of object id numbers
        """
        if frame is None:
            ids = []
            for frame in self.frames:
                ids += frame.get_ids()
            return ids
        else:
            return self.frames[frame].get_ids()

    def get_boxes(self, frame):
        """
        :param frame: int: index to a frame in self.frames
        :return: np.array
        """
        return self.frames[frame].get_boxes()

    def get_conf(self, index):
        """
        :param index: int: index to a frame in self.frames
        :return: np.array
        """
        return self.frames[index].get_conf()

    def get_appearances(self, frame=None, shape=None):
        """
        :param frame: int: index to a frame in self.frames
        :param shape: tuple: size of the appearances to be returned
        :return: np.array
        """
        if frame is not None:
            return self.frames[frame].get_appearances(shape=shape)
        else:
            print("Extracting appearances from %s" % self.info.name)
            appearances = []
            progress_bar = Progbar(self.get_n_frames(), width=30, verbose=1, interval=1)
            for i, frame in enumerate(self.frames):
                if shape is None:
                    appearances += frame.get_appearances(shape=shape)
                else:
                    appearances.append(frame.get_appearances(shape=shape))
                progress_bar.update(i)
            print("\n")
            if shape is not None:
                appearances = np.concatenate(appearances)
        return appearances

    def show(self, scale=1, width=1, draw=True):
        """
        :return: None
        """
        clock = Clock(self.info.frameRate)
        for frame in self.frames:
            image = frame.get_frame(width=width, scale=scale, draw=draw)
            cv2.imshow(self.name, image)
            cv2.waitKey(1)
            clock.toc()

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
