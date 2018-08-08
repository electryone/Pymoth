#!/usr/bin/enb python3


class Namespace:

    def __init__(self, dictionary=None):
        if kwargs is not None:
            self.__dict__.update(dictionary)

    def add(self, dictionary):
        """
        :param dictionary:
        :return:
        """
        self.__dict__.update(dictionary)

    def get(self):
        """
        :return:
        """
        return self.__dict__

    def summary(self, tab_size=2, tabs=0):
        """
        :param tab_size:
        :param tabs:
        :return:
        """
        for key, item in self.__dict__.items():
            if isinstance(item, Namespace):
                print("%s%s:" % (" " * tab_size * tabs, key))
                item.summary(tabs=tabs + 1)
            else:
                print("%s%s: %s" % (" " * tab_size * tabs, key, item))