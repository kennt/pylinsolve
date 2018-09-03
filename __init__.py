""" Root module for pysolve tool.

    Copyright (c) 2014 Kenn Takara
    See LICENSE for details

"""


class InvalidNameError(ValueError):
    """ Exception: Invalid name. """
    def __init__(self, name, text):
        super(InvalidNameError, self).__init__()
        self.text = "invalid name : {0} : {1}".format(name, text)
        self.name = name

    def __str__(self):
        return self.text
