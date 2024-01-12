MultiVariate Viz
================

Copyright 2024, Hilary Luo and contributors

Licensed under LGPL v3 

A simple python GUI tool to visualize multivariate data and create PCA models.
Tested on Windows and Ubuntu.

Icons sourced from https://p.yusukekamiyamane.com/, created by Yusuke Kamiyamane
and distributed under Creative Commons Attribution 3.0 License.

Requirements
------------

* Python 3.9+
* numpy 1.22+
* pandas 1.5+
* scikit-learn 1.2.2+
* PyQt6 6.4+
* pyqtgraph 0.13.3+

Installation
------------

* Using PyPI (if Git is installed):

  * `pip install git+https://github.com/hilary-luo/mv_viz.git`

* Using PyPI (if Git is *not* installed): 

  * `pip install https://github.com/hilary-luo/mv_viz/tarball/main`

* From source distribution:

  * Download the repository or run 
  `git clone https://github.com/hilary-luo/mv_viz.git`

  * Then run `pip install .` or `python setup.py install` from the directory 
  where you download this repository.

Usage
-----

* Once installed, the python setup should automatically create a console 
script wrapper that can be run from command line or terminal anywhere by 
simply running `mv_viz` 

  * Note: This may fail if the python console scripts installation directory 
  path is not inculded in the PATH environment variable.

* Alternatively, run `python src/mv_viz/mv_viz.py` from the directory where 
you download this repository. This option does not require installation.

Documentation
-------------

Under development