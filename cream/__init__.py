import time
__import__start__ = time.time()

# import section

from cream.engine.deeplearning import network

import cream.functions as functions

import cream.layer as layer

import cream.tool.csys as csys
import cream.tool.colors as colors

# result section

__help__ = """ CREAM Version 1.0.0 by @mangto  (github)\nBug Report/Contact: 'mangto0701@gmail.com'"""
__import__end__ = time.time()
__import__estimated__ = __import__end__ - __import__start__

print(__help__)
print(f"{'Loading Estimated: ' + str(round(__import__estimated__, 3)) + 's':^42}")
csys.division(42)