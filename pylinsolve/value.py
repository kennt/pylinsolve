""" Contains the base class for Variables and Parameters/

    Copyright (c) 2014 Kenn Takara
    See LICENSE for details

"""


class Value(object):
	def __init__(self, name, desc=None, default=None, symbol=None):
		self.name = name
		self.desc = desc
		self.default = default

		self.symbol = symbol or sympy.Symbol(name)

