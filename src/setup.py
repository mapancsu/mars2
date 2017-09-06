'''
Created on 2013-9-26

@author: humblercoder
'''

from distutils.core import setup
import py2exe, sys
import matplotlib

sys.path.append('D:\python2.7.10\Lib\site-packages\PyQt4') 

opts = {'py2exe': {"includes": ["FileDialog","matplotlib.backends.backend_tkagg",
						"scipy.special._ufuncs_cxx",
						"scipy.sparse.csgraph._validation",
						"scipy.io.matlab.streams",
						"scipy.linalg.cython_blas",
						"scipy.linalg.cython_lapack",
						"scipy.integrate"],
			 'dll_excludes': [r'MSVCP90.dll'],
				   }
		}
mars_app ={"script": "MARS.pyw"}

setup(windows=[mars_app], data_files=matplotlib.get_py2exe_datafiles(), options=opts)
