#!/usr/bin/enb python3


class Namespace:

    def __init__(self, dictionary=None):
        if dictionary is not None:
            self.__dict__.update(dictionary)

    def add(self, dictionary):
        """
        :param dictionary:
        :return:
        """
        self.__dict__.update(dictionary)

    def get(self, set_name=None, namespace=None):
        """
        :param set_name:
        :param namespace:
        :return:
        """
        if set_name is None:
            return self.__dict__
        else:
            sub_set = []
            namespace = self if namespace is None else namespace
            for key, item in namespace.get().items():
                if key == set_name:
                    sub_set.append(item)
                elif isinstance(item, Namespace):
                    sub_set += self.get(set_name, item)
            return sub_set

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
