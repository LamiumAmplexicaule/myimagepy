from distutils.core import setup
from distutils.extension import Extension

import numpy as np

if __name__ == "__main__":
    setup(name='myimage',
          version="1.0.0",
          ext_modules=[
              Extension('myimage',
                        ['myimage.c'],
                        include_dirs=[np.get_include()],
                        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
                        ),
          ])
