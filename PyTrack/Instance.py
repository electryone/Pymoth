#!/usr/bin/env python3


"""
An Instance object is used to represent an object at a single frame in time.
An Instance can be defined using a bounding box or world coordinates.
"""


import cv2
import numpy as np

from PyTrack.utils import resize
from PyTrack.utils import box2rect
from PyTrack.utils import box2xywh


class Instance(object):

    def __init__(self, id=-1, img_path=None, frame_index=None, bounding_box=None, coordinates=None, conf=None):
        self.id = id                                        # Identification number of instance
        self.frame_index = frame_index
        self.img_path = img_path
        self.bounding_box = bounding_box                    # Bounding box of instance
        self.conf = conf
        self.coordinates = coordinates                      # World coordinates of instance
        self.mode = None                                    # Either "bounding_box" or "world_coordinates"
        if bounding_box is not None:
            self.__set_bounding_box()                       # Set mode and cast bounding_box as np array
        if coordinates is not None:
            self.__set_coordinates()                        # Set mode and cast coordinates as np array
        if id != -1:
            np.random.seed(id)
            self.color = tuple(map(int, np.random.randint(0, 255, 3)))  # Set a random color seeded by the instance id
        else:
            self.color = (255, 255, 255)

    def __set_bounding_box(self):
        """
        Cast coordinates as np.array and set mode
        :return: None
        """
        self.bounding_box = np.asarray(self.bounding_box)
        self.mode = "bounding_box"

    def __set_coordinates(self):
        """
        Cast coordinates as np.array and set mode
        :return: None
        """
        self.coordinates = np.asarray(self.coordinates)
        self.mode = "world_coordinates"

    def get_box(self):
        """
        :return:
        """
        return self.bounding_box

    def get_rect(self):
        """
        :return:
        """
        return box2rect(self.bounding_box)

    def get_xywh(self):
        """
        :return:
        """
        return box2xywh(self.bounding_box)

    def get_conf(self):
        """
        :return:
        """
        return self.conf

    def get_id(self):
        """
        :return:
        """
        return self.id

    def get_image_path(self):
        """
        :return:
        """
        return self.img_path

    def get_frame_index(self):
        """
        :return:
        """
        return self.frame_index

    def get_mode(self):
        """
        :return:
        """
        return self.mode

    def get_color(self):
        """
        :return:
        """
        return self.color

    def set_color(self, color):
        """
        :param color:
        :return:
        """
        self.color = color

    def get_appearance(self, shape=None):
        if self.mode == "bounding_box":
            rect = self.get_rect()
            rect[rect < 0] = 0
            x0, y0, x1, y1 = rect
            if shape is None:
                return cv2.imread(self.img_path)[y0:y1, x0:x1]
            else:
                return resize(cv2.imread(self.img_path)[y0:y1, x0:x1], shape)
        else:
            raise NotImplementedError("Get appearance not yet implemented for world_coordinates mode")

    def draw(self, image, width=1, scale=1, show_ids=False):
        """
        :param image:
        :param width:
        :param scale:
        :param show_ids:
        :return:
        """
        if self.mode == "bounding_box":
            rect = self.get_rect() * scale
            x1, y1, x2, y2 = rect.astype(int)
            image = cv2.rectangle(image, pt1=(x1, y1), pt2=(x2, y2), color=self.color, thickness=width)
            if show_ids:
                image = cv2.putText(image, "%s" % self.id, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 2 * scale, self.color)
            return image
        elif self.mode == "world_coordinates":
            print("Warning: Instance.draw() is not yet implemented for 'world_coodrinates'")
            return image
        else:
            print("Warning: unknown instance mode, %s" % self.mode)
            return image
