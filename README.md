# 1. Intorduction #

*MS-assisted resolution of signal  (MARS) is a powerful approach for extractin features from large-scale GC-MS datasets. Classical multivarite resolution methods were integrated and user interface was used to tune parameters and visualize results. MARS2 has improved both the robustness and sensitivity of extracted features.*


![graphical abstract of mars](https://raw.githubusercontent.com/mapancsu/mars2/master/MARS2.jpg)

# 2. Installation #


## 2.1 Python version ##

MARS2 was implemented efficiently in Python programming language. All resource code were available at [url](https://github.com/mapancsu/mars2/tree/master/src). Follow the instructions below to compile, debug and run mars2.

- Install Python
	Python 2.7 is recommended
	https://www.python.org/ftp/python/2.7.10/python-2.7.10.msi


- Install Numpy, Scipy, Matplotlib, PyQt4 with following commands 

	```shell
	pip install numpy
	pip install scipy
	pip install matplotlib
    pip install PyQt4
	```
- clone this project and run [main.py](https://github.com/mapancsu/mars2/tree/master/src/main.py)

## 2.2 Executable version ##

We have already noticed that user interface for feature extraction was helpful for parameter optimizing problem and stability of results. So we have compiled this MARS2 algorithm and encapsulate with inno setup to provide a better user interface for feature extraction. One can tune  parameters and visualize results easily.

- It can be downloaded from this [url](https://github.com/mapancsu/mars2/tree/master/win64.exe)
- Click the downloaded exe file, install it in your computer and run the installed program.

## 2.3 Guide for user ##

MARS2 include two modules (Resolve and Extract). All details of resolving mass spectral and retention time (MSRT) pairs, and extracting features from batch datasets are available at [url](https://github.com/mapancsu/mars2/blob/master/guide%20for%20user.pdf)

# 3. Contact #

For any questions, please contact:

[Mapan_csu@yahoo.com](mailto:Mapan_csu@yahoo.com)
