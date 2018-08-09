#!/usr/bin/env python3


"""
Author: Samuel Westlake
E-mai: s.t.westlake@cranfield.ac.uk

An Instance object is used to represent an object at a single frame in time.
An Instance can be defined using a bounding box or world coordinates.
WARNING: Full functionality for the world coordinate system is not yet implemented.
"""


import cv2
import numpy as np

from pymoth.utils import resize
from pymoth.utils import box2rect
from pymoth.utils import box2xywh


class Instance(object):

    def __init__(self, id_number=-1,
                 img_path=None,
                 frame_index=None,
                 bounding_box=None,
                 coordinates=None,
                 conf=None,
                 state=None,
                 color=None):
        """
        :param id_number: int: the unique identification number of the instance
        :param img_path: str: the path to the image that contains the instance
        :param frame_index: int: the index number of the frame that contains the instance
        :param bounding_box: np.array(1, 4): the bounding box of the instance (left, top, width, height)
        :param coordinates: np.array(1, 3): the world coordinates of the instance (...)
        :param conf: int: the detection confidence of the instance (default = -1)
        :param state: str: the human-readable state of the instance
        :param color: tuple: the color used when drawing the instance bounding box
        """
        # Private variables
        self._bounding_box = None
        self._coordinates = None
        self._id = None
        # Public variables
        self.color = color
        self.conf = conf
        self.frame_index = frame_index
        self.img_path = img_path
        self.mode = None
        self.state = state
        # Set bounding_box / coordinates and id
        if bounding_box is not None:
            self.set_bounding_box(bounding_box)
        if coordinates is not None:
            self.set_coordinates(coordinates)
        self.set_id(id_number)

    def set_bounding_box(self, bounding_box):
        """
        Cast coordinates as np.array and set mode
        :return: None
        """
        self._bounding_box = np.asarray(bounding_box)
        self.mode = "bounding_box"

    def set_coordinates(self, coordinates):
        """
        Cast coordinates as np.array and set mode
        :return: None
        """
        self._coordinates = np.asarray(coordinates)
        self.mode = "world_coordinates"

    def set_id(self, id_number):
        """
        Sets the instance id
        If the instnace color is not already set, sets the instance color based on the instance id number
        :param id_number: int: the unique identification number of the instance
        :return: None
        """
        self._id = id_number
        if self._id == -1:
            self.color = (255, 255, 255)
        else:
            np.random.seed(id_number)
            self.color = tuple(map(int, np.random.randint(0, 255, 3)))

    def get_bounding_box(self):
        """
        :return: np.array(1, 4): the instance bounding box (left, top, width, height)
        """
        return self._bounding_box

    def get_rect(self):
        """
        :return: np.array(1, 4): the instance rect (left, top, right, bottom)
        """
        return box2rect(self._bounding_box)

    def get_state(self):
        """
        :return: str: the human-readable state of the instance
        """
        return self.state

    def get_xywh(self):
        """
        :return: np.array(1, 4): the bounding box defined by (left, top, width, height)
        """
        return box2xywh(self._bounding_box)

    def get_id(self):
        """
        :return: int: the unique identification number of the instance
        """
        return self._id

    def get_appearance(self, shape=None, keep_aspect=True):
        """
        :param shape: the required shape of the output image
        :param keep_aspect: bool: whether to keep the object aspect ratio or not when resizing
        :return: np.array: the image of the instance
        """
        if self.mode == "bounding_box":
            rect = self.get_rect()
            rect[rect < 0] = 0
            x0, y0, x1, y1 = rect
            if shape is None:
                return cv2.imread(self.img_path)[y0:y1, x0:x1]
            else:
                if keep_aspect:
                    return resize(cv2.imread(self.img_path)[y0:y1, x0:x1], shape)
                else:
                    return cv2.resize(cv2.imread(self.img_path)[y0:y1, x0:x1], shape[0: 2])
        else:
            raise NotImplementedError("Get appearance not yet implemented for world_coordinates mode")

    def show(self, image=None, draw=False, width=1, scale=1, show_ids=False):
        """
        :param image: np.array: the image on which to draw the instance
        :param draw: bool: whether or not to draw the instance bounding box / world coordinates
        :param width: int: the line width of the instance bounding box
        :param scale: int: the scale of the drawing
        :param show_ids: bool: whether or not to draw the instance id number
        :return: np.array: the original image with the instance drawn
        """
        if image is None:
            image = cv2.imread(self.img_path)
        if draw:
            if self.mode == "bounding_box":
                rect = self.get_rect() * scale
                x1, y1, x2, y2 = rect.astype(int)
                image = cv2.rectangle(image, pt1=(x1, y1), pt2=(x2, y2), color=self.color, thickness=width)
                if show_ids and self.get_id() != -1:
                    image = cv2.putText(image, "%s" % self._id, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 2 * scale, self.color, thickness=width)
                return image
            elif self.mode == "world_coordinates":
                raise NotImplementedError("Instance.draw() is not yet implemented for 'world_coordinates'")
            else:
                raise ValueError("Unknown instance mode, %s" % self.mode)
