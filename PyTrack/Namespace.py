#!/usr/bin/enb python3


class Namespace:

    def __init__(self, dictionary=None):
        if dictionary is not None:
            self.__dict__.update(dictionary)

    def add(self, dictionary, sub_space=None):
        """
        :param dictionary:
        :param sub_space:
        :return:
        """
        if sub_space is None:
            self.__dict__.update(dictionary)
        else:
            if isinstance(sub_space, list):
                sub_space = sub_space[0] if len(sub_space) == 1 else sub_space
            if isinstance(sub_space, list):
                self.get()[sub_space[0]].add(dictionary, sub_space[1:])
            else:
                self.get()[sub_space].add(dictionary)

    def get(self, key=None):
        """
        Note: key can be a list of sub Namespace names for which get() will be called recursively
        :param key: key of the sub item to be returned
        :return: dict
        """
        if key is None:
            return self.__dict__
        else:
            if isinstance(key, list):
                key = key[0] if len(key) == 1 else key
            if isinstance(key, list):
                return self.get(key[0]).get(key=key[1:])
            else:
                return self.__dict__[key]

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
