import time
__import__start__ = time.time()

# import section

from cream.engine.deeplearning import network
import cream.engine.word2vec as word2vec

import cream.functions as functions

import cream.layer as layer

import cream.tool.csys as csys
import cream.tool.colors as colors
import cream.tool.datasets as datasets
import cream.tool.progress_bar as progress_bar

import cream.generator as generator


import cream.trainer as trainer

# result section
__version__ = '1.1.0'
__all__ = [
    'network', 'word2vec',
    'functions', 'generator',
    'layer', 'csys', 'colors', 'datasets', 'progress_bar', 
    'trainer'
    ]
__help__ = f""" CREAM Version {__version__} by @mangto  (github)\nBug Report/Contact: 'mangto0701@gmail.com'"""
__import__end__ = time.time()
__import__estimated__ = __import__end__ - __import__start__

print(__help__)
print(f"{'Loading Estimated: ' + str(round(__import__estimated__, 3)) + 's':^42}")
csys.division(42)