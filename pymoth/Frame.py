#!/usr/bin/env python3

import cv2
import numpy as np

from pymoth.Instance import Instance
from pymoth.utils import resize
from pymoth.utils import box2xywh
from pymoth.utils import box2rect


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

    def get_image(self, width=1, scale=1, draw=False, show_ids=False, states=None):
        """
        :param width:
        :param scale:
        :param draw:
        :param show_ids:
        :param states:
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
            for instance in self.get_instances(states=states):
                image = instance.show(image=image, draw=draw, width=width, scale=scale, show_ids=show_ids)
        return image

    def get_n_instances(self, id=None):
        """
        :return: int: the number of instances in the frame
        """
        if id is None:
            return len(self.instances)
        else:
            return len([1 for instance in self.instances if instance.get_id() == id])

    def get_instances(self, index=None, id=None, states=None):
        """
        :param index:
        :param id:
        :param states:
        :return:
        """
        # Get all instances
        if index is None and id is None and states is None:
            return self.instances
        # Get instances with given state
        elif states is not None and id is None and index is None:
            return [instance for instance in self.instances if instance.get_state() in states]
        # Get instances with given id
        elif id is not None and states is None and index is None:
            return [instance for instance in self.instances if instance.get_id() == id]
        # Get instance with given index
        elif index is not None and states is None and id is None:
            return self.instances[index]
        # Get instance with given id and states
        elif id is not None and states is not None and index is None:
            return [instance for instance in self.instances if instance.get_id() == id and instance.get_state() in states]
        else:
            raise NotImplementedError("combination of kwargs is currently not supported")

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

    def get_boxes(self, id=None):
        """
        :param id:
        :return:
        """
        bounding_boxes = np.empty((self.get_n_instances(id=id), 4))
        for i, instance in enumerate(self.get_instances(id=id)):
            bounding_boxes[i] = instance.get_bounding_box()
        return bounding_boxes

    def get_xywh(self, id=None):
        """
        :param id:
        :return:
        """
        return box2xywh(self.get_boxes(id=id))

    def get_rects(self, id=None):
        """
        :param id:
        :return:
        """
        return box2rect(self.get_boxes(id=id))

    def get_conf(self):
        """
        :return: list: confidence values for each instance in the frame
        """
        return [instance.conf for instance in self.instances]

    def get_appearances(self, id=None, shape=None):
        """
        :param id:
        :param shape:
        :return:
        """
        image = cv2.imread(self.img_path)
        rects = self.get_rects(id=id).astype(int)
        rects[rects < 0] = 0
        if shape is None:
            appearances = []
            for x0, y0, x1, y1 in rects:
                appearances.append(image[y0:y1, x0:x1])
        else:
            appearances = np.empty((tuple([self.get_n_instances(id=id)] + list(shape))), dtype=np.uint8)
            for i, (x0, y0, x1, y1) in enumerate(rects):
                appearances[i] = resize(image[y0:y1, x0:x1], shape)
        return appearances
