from distutils.core import setup
from Cython.Build import cythonize
from shutil import move

setup(
    ext_modules = cythonize("./Z_Model*.pyx")
)

# get that shit outta the unecessary 'zagaris_model/zagaris_model' directory
move('./zagaris_model/Z_Model.so', './Z_Model.so')
move('./zagaris_model/Z_Model_Transformed.so', './Z_Model_Transformed.so')
