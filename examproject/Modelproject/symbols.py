# -*- coding: utf-8 -*-
"""
Created 19th may 2019

@author: pbj568
"""

import sympy as sm

# define all symbols to basic model
c = sm.symbols('c', positive = True)
e = sm.symbols('e')
y = sm.symbols('y')
w = sm.symbols('w')
b = sm.symbols('b')
s = sm.symbols('s')

# define additional symbols to extended model
sigma = sm.symbols('sigma^2', positive = True)
r = sm.symbols('r', positive = True)