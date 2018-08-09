#!/usr/bin/enb python3

import collections
import cv2
import numpy as np
import sys
import time


from pymoth.Namespace import Namespace

"""
chunks(l, n)
get_shifty(array, shift=1)
convert(string)
load_info(file_path)
resize(image, shape, keep_aspect=True, padding=0)
pad(image, shape, value=0)
box2rect(box)
rect2box(rect)
box2xywh(box)
iou(rects)
iou2(rects1, rects2)
nms(array, by_row=True, by_col=True, threshold=0)
Progbar(target, width=30, verbose=1, interval=0.05, stateful_metrics=None)
"""


def chunks(l, n):
    """
    :param l:
    :param n:
    :return:
    """
    n = max(1, n)
    return list(l[i:i+n] for i in range(0, len(l), n))


def get_shifty(array, shift=1):
    """
    :param array:
    :param shift:
    :return:
    """
    return array[:, -shift], array[:, shift:]


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
            if "=" in line:
                var, value = line.split("=")
                info.add({var: convert(value)})
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
    xywh = np.empty_like(box.T)
    xywh[0] = box.T[0] + box.T[2] / 2
    xywh[1] = box.T[1] + box.T[3] / 2
    xywh[2] = box.T[2]
    xywh[3] = box.T[3]
    return xywh.T


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


def iou2(rects1, rects2):
    """
    :param rects1:
    :param rects2:
    :return:
    """
    mat = np.empty((len(rects1), len(rects2)))
    for j, r1 in enumerate(rects1):
        for i, r2 in enumerate(rects2):
            x0 = max(r2[0], r1[0])
            y0 = max(r2[1], r1[1])
            x1 = min(r2[2], r1[2])
            y1 = min(r2[3], r1[3])
            if x0 < x1 and y0 < y1:
                inter_area = (x1 - x0) * (y1 - y0)
            else:
                inter_area = 0
            i_area = (r2[2] - r2[0]) * (r2[3] - r2[1])
            j_area = (r1[2] - r1[0]) * (r1[3] - r1[1])
            mat[j, i] = inter_area / (i_area + j_area - inter_area)
    return mat


def nms(array, by_row=True, by_col=True, threshold=0):
    """
    :param array:
    :param by_row:
    :param by_col:
    :param threshold:
    :return:
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
    if threshold:
        array[array < threshold] = 0
    return array


class Progbar(object):
    """Displays a progress bar.

    # Arguments
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over time. Metrics in this list
            will be displayed as-is. All others will be averaged
            by the progbar before display.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1, interval=0.05,
                 stateful_metrics=None):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        self._values = collections.OrderedDict()
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None):
        """Updates the progress bar.

        # Arguments
            current: Index of current step.
            values: List of tuples:
                `(name, value_for_last_step)`.
                If `name` is in `stateful_metrics`,
                `value_for_last_step` will be displayed as-is.
                Else, an average of the metric over time will be displayed.
        """
        values = values or []
        for k, v in values:
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                # Stateful metrics output a numeric value.  This representation
                # means "take an average from a single value" but keeps the
                # numeric formatting.
                self._values[k] = [v, 1]
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval and
                    self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600, (eta % 3600) // 60, eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k in self._values:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(
                        self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self._values:
                    info += ' - %s:' % k
                    avg = np.mean(
                        self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)
