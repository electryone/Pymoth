#!/usr/bin/env python3

import time


class Clock(object):

    def __init__(self, frame_rate=25):
        self.t0 = 0
        self.wait_time = 0
        self.frame_rate = frame_rate
        self.delay = 1 / frame_rate

    def toc(self):
        self.wait_time = time.time() - self.t0              # Time passed since last toc
        while self.wait_time < self.delay:                  # While wait time is less than delay
            self.wait_time = time.time() - self.t0          # Time passed since last toc
        self.t0 = time.time()                               # Set time of last toc to current time


