#!/usr/bin/enb python3


class Namespace:

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def add(self, **kwargs):
        self.__dict__.update(kwargs)

    def summary(self, tabs=0):
        for key, item in vars(self).items():
            if isinstance(item, Namespace):
                print("%s%s:" % ("  " * tabs, key))
                item.summary(tabs=tabs + 1)
            else:
                print("%s%s: %s" % ("  " * tabs, key, item))
