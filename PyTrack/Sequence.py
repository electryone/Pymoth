#!/usr/bin/env python3

"""
A Sequence object contains a list of frames, each frame has a list of instances (each with an id number).
"""

import os
import cv2
import copy
import numpy as np
from itertools import count
from keras.utils.generic_utils import Progbar

from PyTrack.Frame import Frame
from PyTrack.Clock import Clock
from PyTrack.utils import convert
from PyTrack.utils import box2rect
from PyTrack.utils import box2xywh

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
        self.img_dir = None
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
                kwargs = {"id_number": data["id"],
                          "bounding_box": (data["bb_left"],
                                           data["bb_top"],
                                           data["bb_width"],
                                           data["bb_height"]),
                          "conf": data["conf"]}
                self.create_instance(data["frame"], kwargs)

    def init_frames(self, info=None, n=None, img_dir=None):
        if info is not None:
            self.info = copy.deepcopy(info)
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
        self.img_dir = img_dir
        frame_paths = os.listdir(self.img_dir)
        for path in frame_paths:
            i = int(path.split(".")[0]) - 1
            self.set_frame_path(i, "%s/%s" % (self.img_dir, path))

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

    def get_frame(self, frame):
        """
        :return:
        """
        return self.frames[frame]

    def get_frames(self):
        """
        :return:
        """
        return self.frames

    def get_n_frames(self):
        """
        :return: int: the number of frames in the sequence
        """
        return len(self.frames)

    def get_n_instances(self, frame=None, id=None):
        """
        :param frame:
        :param id:
        :return:
        """
        if frame is None:
            if id is None:
                return len([1 for frame in self.frames for _ in frame.instances])
            else:
                return len([1 for frame in self.frames for instance in frame.instances if instance.id == id])
        else:
            if not 0 <= frame < self.get_n_frames():
                return 0
            if id is None:
                return len([1 for _ in self.frames[frame].instances])
            else:
                return len([1 for instance in self.frames[frame].instances if instance.id == id])

    def get_instances(self, frame=None, id=None):
        """
        :param frame:
        :param id:
        :return:
        """
        if frame is None:
            if id is None:
                return [instance for frame in self.frames for instance in frame.instances]
            else:
                return [instance for frame in self.frames for instance in frame.instances if instance.id == id]
        else:
            if not 0 <= frame < self.get_n_frames():
                return []
            if id is None:
                return [instance for instance in self.frames[frame].instances]
            else:
                return [instance for instance in self.frames[frame] if instance.id == id]

    def get_ids(self, frame=None):
        """
        :return:
        """
        if frame is None:
            return [instance.id for frame in self.frames for instance in frame.instances]
        else:
            return [instance.id for instance in self.frames[frame].instances]

    def get_n_ids(self, frame=None):
        """
        :return: int: the number of target in the frame or sequence
        """
        ids = np.unique(self.get_ids(frame=frame))
        return len(np.delete(ids, np.where(ids == -1)))

    def get_unique_ids(self):
        """
        :return:
        """
        return np.unique(self.get_ids())

    def get_boxes(self, frame=None, id=None):
        """
        :param frame:
        :param id:
        :return:
        """
        boxes = np.empty((self.get_n_instances(frame=frame, id=id), 4))
        for i, instance in enumerate(self.get_instances(frame=frame, id=id)):
            boxes[i] = instance.get_box()
        return boxes

    def get_rects(self, frame=None, id=None):
        """
        :param frame:
        :param id:
        :return:
        """
        return box2rect(self.get_boxes(frame=frame, id=id))

    def get_xywh(self, frame=None, id=None):
        """
        :param frame:
        :param id:
        :return:
        """
        return box2xywh(self.get_boxes(frame=frame, id=id))

    def get_conf(self, frame=None, id=None):
        """
        :param frame:
        :param id:
        :return:
        """
        if frame is None:
            if id is None:
                return [instance.conf for frame in self.frames for instance in frame.instances]
            else:
                return [instance.conf for frame in self.frames for instance in frame.instances if instance.id == id]
        else:
            if id is None:
                return [instance.conf for instance in self.frames[frame].instances]
            else:
                return [instance.conf for instance in self.frames[frame].instances if instance.id == id]

    def get_appearances_by_id(self, shape=None):
        """
        :param shape:
        :return:
        """
        print("Getting the appearances from %s frames" % self.get_n_frames())
        if shape is None:
            appearances = [[] for _ in self.get_unique_ids()]
            progress_bar = Progbar(self.get_n_frames(), width=30, verbose=1, interval=1)
            for i, frame in enumerate(self.frames):
                for appearance, id in zip(frame.get_appearances(shape=shape), frame.get_ids()):
                    appearances[id - 1].append(appearance)
                progress_bar.update(i)
        else:
            appearances = [np.empty(tuple([self.get_n_instances(id=id)] + list(shape)), dtype=np.uint8)
                           for id in self.get_unique_ids()]
            indexes = [count() for _ in self.get_unique_ids()]
            progress_bar = Progbar(self.get_n_frames(), width=30, verbose=1, interval=1)
            for i, frame in enumerate(self.frames):
                for appearance, id in zip(frame.get_appearances(shape=shape), frame.get_ids()):
                    appearances[id - 1][next(indexes[id - 1])] = appearance
                progress_bar.update(i)
        print("\n")
        return appearances

    def get_appearances(self, frame=None, id=None, shape=None):
        """
        :param id:
        :param shape:
        :return:
        """
        if frame is None:
            return self.__from_all_frames(id=id, shape=shape)
        if frame is not None:
            return self.__from_frame(frame, id=id, shape=shape)

    def __from_frame(self, frame, id=None, shape=None):
        if not 0 <= frame < self.get_n_frames():
            if shape is None:
                return []
            else:
                return np.empty((tuple([0] + list(shape))))
        else:
            return self.frames[frame].get_appearances(id=id, shape=shape)

    def __from_all_frames(self, id=None, shape=None):
        print("Getting the appearances from %s frames" % self.get_n_frames())
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

    def show(self, frame=None, scale=1, width=1, draw=False, show_ids=False, restrict_fps=True, states=None):
        """
        :param frame:
        :param scale:
        :param width:
        :param draw:
        :param show_ids:
        :param restrict_fps:
        :param states:
        :return:
        """
        if frame is None:
            clock = Clock(self.info.frameRate)
            for frame in self.frames:
                image = frame.get_image(width=width, scale=scale, draw=draw, show_ids=show_ids, states=states)
                cv2.imshow(self.info.name, image)
                cv2.waitKey(1)
                if restrict_fps:
                    clock.toc()
        else:
            image = self.frames[frame].get_image(width=width, scale=scale, draw=draw, show_ids=show_ids, states=states)
            cv2.imshow(self.info.name, image)
            cv2.waitKey(1)

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
