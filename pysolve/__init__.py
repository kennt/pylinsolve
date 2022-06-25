""" Root module for pysolve tool.

    Copyright (c) 2014-2022 Kenn Takara
    See LICENSE for details

"""


class InvalidNameError(ValueError):
    """ Exception: Invalid name. """
    def __init__(self, name, text):
        super().__init__()
        self.text = f"invalid name : {name} : {text}"
        self.name = name

    def __str__(self):
        return self.text
