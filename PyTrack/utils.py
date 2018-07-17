#!/usr/bin/enb python3

import cv2
import numpy as np
from itertools import tee

from .Namespace import Namespace


def convert(string):
    """
    :param string: str: any string
    :return: input variable as int or float or string (whichever is appropriate)
    """
    try:
        return int(string)
    except ValueError:
        try:
            return float(string)
        except ValueError:
            return "%s" % string


def pairwise(iterable):
    """
    :param iterable:
    :return:
    """
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def load_info(file_path):
    """
    Load info file into the info namespace
    :param file_path: Path to an info file
    :return: A name space containing information form the file
    """
    info = Namespace()
    with open(file_path, "r") as file:
        for line in file:
            line = line.replace("\n", "")
            try:
                var = line.split("=")[0]
                value = convert(line.split("=")[1])
                if isinstance(value, str):
                    value = "\'%s\'" % value
                exec("info.add(%s=%s)" % (var, value))
            except IndexError:
                pass
    return info


def resize(image, shape, keep_aspect=True, padding=0):
    """
    Author: Samuel Westlake and Alix Leroy
    :param image: np.array, input image
    :param shape: tuple, target shape
    :param keep_aspect: bool, whether or not the aspect ration should be kept
    :param padding: int, value for padding if keep_aspect is True
    :return: np.array, image of size shape
    """
    if image.shape[0]*image.shape[1] > shape[0]*shape[1]:
        interpolation = cv2.INTER_LINEAR_EXACT                          # Use the Bilinear Interpolation
    else:
        interpolation = cv2.INTER_CUBIC                                 # Use the Bicubic interpolation
    if keep_aspect:
        scale = min(np.asarray(shape[0:2]) / np.asarray(image.shape[0:2]))
        new_size = np.array(image.shape[0:2]) * scale
        image = cv2.resize(image, (int(new_size[1]), int(new_size[0])), interpolation=interpolation)
        image = pad(image, shape, padding)
    else:
        image = cv2.resize(image, (shape[0], shape[1]), interpolation=interpolation)
    return image


def pad(image, shape, value=0):
    """
    Author: Samuel Westlake and Alix Leroy
    Pads an image to self.x_size with a given value with the image centred
    :param: image: input image
    :param: value
    :return: Padded image
    """
    padded = np.empty(shape, dtype=np.uint8)
    padded.fill(value)
    y0 = int((shape[0] - image.shape[0]) / 2)
    x0 = int((shape[1] - image.shape[1]) / 2)
    y1 = y0 + image.shape[0]
    x1 = x0 + image.shape[1]
    padded[y0:y1, x0:x1, :] = image
    return padded


def iou(rects):
    """
    :param rects: np.array: 2D array of rects (x1, y1, x2, y2)
    :return: np.array: iou distances for each pair of rects
    """
    n = rects.shape[0]
    iou = np.empty((n, n))
    for j in range(n):
        for i in range(n):
            x0 = max(rects[i, 0], rects[j, 0])
            y0 = max(rects[i, 1], rects[j, 1])
            x1 = min(rects[i, 2], rects[j, 2])
            y1 = min(rects[i, 3], rects[j, 3])
            if x0 < x1 and y0 < y1:
                inter_area = (x1 - x0) * (y1 - y0)
            else:
                inter_area = 0
            i_area = (rects[i, 2] - rects[i, 0]) * (rects[i, 3] - rects[i, 1])
            j_area = (rects[j, 2] - rects[j, 0]) * (rects[j, 3] - rects[j, 1])
            iou[j, i] = inter_area / (i_area + j_area - inter_area)
    return iou


def box2rect(box):
    """
    :param box: np.array: array of boxes (left, top, w, h) can be 1D or 2D
    :return: rect: np.array: 2D array of rects (x1, y1, x2, y2)
    """
    rect = np.empty_like(box.T)
    rect[0] = box.T[0]
    rect[1] = box.T[1]
    rect[2] = box.T[0] + box.T[2]
    rect[3] = box.T[1] + box.T[3]
    return rect.T


def rect2box(rect):
    """
    :param rect: np.array: array of rects (x1, y1, x2, y2) can be 1D or 2D
    :return: box: np.array: np.array: array of boxes (left, top, w, h)
    """
    print("Warning: rect2box is untested")
    box = np.empty_like(rect.T)
    box[0] = rect.T[0]
    box[1] = rect.T[1]
    box[2] = rect.T[2] - rect.T[0]
    box[3] = rect.T[3] - rect.T[1]
    return box.T


def box2xywh(box):
    """
    :param box:
    :return:
    """
    box = np.empty_like(box.T)
    box[0] = box.T[0] + box.T[2] / 2
    box[1] = box.T[1] + box.T[3] / 2
    box[2] = box.T[2]
    box[3] = box.T[3]
    return box.T


def nms(array, by_row=False, by_col=True):
    """
    :param array: np.array: 2D array
    :param by_row: bool: whether to do nms by row
    :param by_col: bool: whether to do nms by column
    :return: np.array: 2D array after non max suppression
    """
    # Apply row and column non maximum suppression
    if by_row:
        for index, row in enumerate(array):
            row[row < np.max(row)] = 0
            array[index, :] = row
    if by_col:
        for index, col in enumerate(array.T):
            col[col < np.max(col)] = 0
            array[:, index] = col
    return array


def print_dict(dictionary, indent=2, level=0):
    """
    :param dictionary:
    :param indent:
    :param level:
    :return:
    """
    space = " " * indent
    for key, item in dictionary.items():
        if isinstance(item, dict):
            print("%s%s:" % (space * level, key))
            print_dict(item, indent=indent, level=level+1)
        elif isinstance(item, list):
            print("%s%s:" % (space * level, key))
            for i in item:
                if isinstance(i, dict):
                    print_dict(i, indent=indent, level=level+1)
                else:
                    print("%s-%s" % (space * (level + 1), i))
        else:
            print("%s%s: %s" % (space * level, key, str(item)))


def create_pairs(x, digit_indices, num_classes):
    """
    Positive and negative pair creation.
    Alternates between positive and negative pairs.
    :param x:
    :param digit_indices:
    :return:
    """
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = np.random.randint(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


