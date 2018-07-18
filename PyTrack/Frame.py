#!/usr/bin/env python3

import cv2
import numpy as np
import time

from .Instance import Instance
from .utils import resize
from .utils import box2xywh
from .utils import box2rect


class Frame(object):

    def __init__(self, index=None, img_path=None):
        self.index = index
        self.img_path = img_path
        self.instances = []

    def create_instance(self, kwargs):
        """
        Create an instance and append the frame
        :param kwargs:
        :return:
        """
        kwargs["img_path"] = self.img_path
        kwargs["frame_index"] = self.index
        self.add_instance(Instance(**kwargs))

    def add_instance(self, instance):
        """
        Appends an instance to self.instances
        :param instance: Instance: an instance object
        :return: None
        """
        instance.img_path = self.img_path
        instance.frame_index = self.index
        self.instances.append(instance)

    def get_image(self, width=1, scale=1, draw=False, show_ids=False):
        """
        :param width:
        :param scale:
        :param draw:
        :param show_ids:
        :return:
        """
        if self.img_path is None:
            raise ValueError("Frame image path not set")
        image = cv2.imread(self.img_path)
        if image is None:
            raise FileNotFoundError("cv2.imread(%s) returned None, check %s" % (self.img_path, self.img_path))
        if scale is not 1:
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        if draw:
            for instance in self.instances:
                image = instance.draw(image, width=width, scale=scale, show_ids=show_ids)
        return image

    def get_n_instances(self):
        """
        :return: int: the number of instances in the frame
        """
        return len(self.instances)

    def get_ids(self):
        """
        :return: list: id values for each instance in the frame
        """
        return [instance.id for instance in self.instances]

    def get_n_ids(self):
        """
        :return: int: the number of target in the frame or sequence
        """
        return len(np.unique(self.get_ids()))

    def get_boxes(self):
        """
        :return: np.array: bounding boxes from a frame (x1, y1, w, h)
        """
        bounding_boxes = np.empty((self.get_n_instances(), 4))
        for i, instance in enumerate(self.instances):
            bounding_boxes[i] = instance.bounding_box
        return bounding_boxes

    def get_xywh(self):
        """
        :return: np.array: xywh values from a frame (x_centre, y_centre, w, h)
        """
        return box2xywh(self.get_boxes())

    def get_rects(self):
        """
        :return: np.array: rectangles from a frame (x1, y1, x2, y2)
        """
        return box2rect(self.get_boxes())

    def get_conf(self):
        """
        :return: list: confidence values for each instance in the frame
        """
        return [instance.conf for instance in self.instances]

    def get_appearances(self, shape=None):
        """
        Loads frame into memory, the frame is then cropped for each instance bounding box
        :param shape:
        :return:
        """
        image = cv2.imread(self.img_path)
        rects = self.get_rects().astype(int)
        rects[rects < 0] = 0
        if shape is None:
            appearances = []
            for x0, y0, x1, y1 in rects:
                appearances.append(image[y0:y1, x0:x1])
        else:
            appearances = np.empty((tuple([self.get_n_instances()] + list(shape))), dtype=np.uint8)
            for i, (x0, y0, x1, y1) in enumerate(rects):
                appearances[i] = resize(image[y0:y1, x0:x1], shape)
        return appearances