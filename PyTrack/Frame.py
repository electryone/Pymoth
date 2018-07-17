#!/usr/bin/env python3

import cv2
import numpy as np
import time

from .Instance import Instance
from .utils import resize
from .utils import box2xywh


class Frame(object):

    def __init__(self, nb=None, img_path=None):
        self.nb = nb
        self.img_path = img_path
        self.instances = []

    def get_image(self, width=1, scale=1, draw=False, shape=None):
        """
        Display a particular frame
        :param width: int: line width
        :param scale: int: scale factor
        :param draw: bool: scalar whether to draw instances or not
        :param shape: tuple: shape of output image
        :return: np.array: image
        """
        if draw is True and shape is not None:
            raise NotImplementedError("Drawing of frames with user selected image shape is not yet implemented")
        if self.img_path is None:
            raise ValueError("Frame image path not set")
        image = cv2.imread(self.img_path)
        if image is None:
            raise FileNotFoundError("cv2.imread(%s) returned None, check %s" % (self.img_path, self.img_path))
        if scale is not 1:
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        if shape is not None:
            image = resize(image, shape)
        if draw:
            for instance in self.instances:
                image = instance.draw(image, width=width, scale=scale)
        return image

    def get_ids(self):
        """
        :return: list: id values for each instance in the frame
        """
        return [instance.id for instance in self.instances]

    def get_boxes(self):
        """
        :return: np.array: bounding boxes from a frame (x1, y1, w, h)
        """
        bounding_boxes = np.empty((len(self.instances), 4))
        for i, instance in enumerate(self.instances):
            bounding_boxes[i, :] = instance.bounding_box
        return bounding_boxes

    def get_xywh(self):
        """
        :return: np.array: xywh values from a frame (x_centre, y_centre, w, h)
        """
        xywh = np.empty((len(self.instances), 4))
        for i, instance in enumerate(self.instances):
            xywh[i, :] = instance.xywh
        return xywh

    def get_rect(self):
        """
        :return: np.array: rectangles from a frame (x1, y1, x2, y2)
        """
        rect = np.empty((len(self.instances), 4))
        for i, instance in enumerate(self.instances):
            rect[i, :] = instance.rect
        return rect

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
        appearances = []
        frame = cv2.imread(self.img_path)
        for instance in self.instances:
            rect = instance.get_rect()
            rect[rect < 0] = 0
            x0, y0, x1, y1 = rect
            if shape is None:
                appearances.append(frame.copy()[y0:y1, x0:x1])
            else:
                appearances.append(resize(frame.copy()[y0:y1, x0:x1], shape))
        if shape is not None:
            appearances = np.stack(appearances)
        return appearances

    def get_n_instances(self):
        """
        :return: int: the number of instances in the frame
        """
        return len(self.instances)

    def create_instance(self, kwargs):
        """
        Create an instance and append the frame
        :param kwargs:
        :return:
        """
        self.add_instance(Instance(**kwargs))

    def add_instance(self, instance):
        """
        Appends an instance to self.instances
        :param instance: Instance: an instance object
        :return: None
        """
        self.instances.append(instance)
