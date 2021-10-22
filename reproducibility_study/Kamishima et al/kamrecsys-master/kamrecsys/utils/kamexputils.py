#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility routines for numerical experiments
"""

from __future__ import (
    print_function,
    division,
    absolute_import,
    unicode_literals)

# =============================================================================
# Imports
# =============================================================================

import logging
import subprocess
import platform

import numpy as np
import scipy as sp
from scipy.sparse import issparse
import sklearn

# =============================================================================
# Metadata variables
# =============================================================================

# =============================================================================
# Public symbols
# =============================================================================

__all__ = []

# =============================================================================
# Constants
# =============================================================================

# =============================================================================
# Variables
# =============================================================================

# =============================================================================
# Functions
# =============================================================================


def json_decodable(x):
    """
    convert to make serializable type

    Parameters
    ----------
    x : dict, list
        container to convert

    .. warning::

        `long` type of Python2 is not supported 
    """
    # import numpy as np
    # from scipy.sparse import issparse

    if isinstance(x, dict):
        for k, v in x.items():
            if isinstance(v, (dict, list)):
                json_decodable(v)
            elif isinstance(v, np.ndarray):
                x[k] = v.tolist()
                json_decodable(x[k])
            elif issparse(v):
                x[k] = v.toarray().tolist()
                json_decodable(x[k])
            elif isinstance(v, np.integer):
                x[k] = int(v)
            elif isinstance(v, np.floating):
                x[k] = float(v)
            elif isinstance(v, np.complexfloating):
                x[k] = complex(v)
            elif not isinstance(v, (bool, int, float, complex)):
                x[k] = str(v)
    elif isinstance(x, list):
        for k, v in enumerate(x):
            if isinstance(v, (dict, list)):
                json_decodable(v)
            elif isinstance(v, np.ndarray):
                x[k] = v.tolist()
                json_decodable(x[k])
            elif issparse(v):
                x[k] = v.toarray().tolist()
                json_decodable(x[k])
            elif isinstance(v, np.integer):
                x[k] = int(v)
            elif isinstance(v, np.floating):
                x[k] = float(v)
            elif isinstance(v, np.complexfloating):
                x[k] = complex(v)
            elif not isinstance(v, (bool, int, float, complex)):
                x[k] = str(v)


def get_system_info(output_node_info=False):
    """
    Get System hardware information

    Parameters
    ----------
    output_node_info : bool, optional
        Include hostname as 'node'.  (default=False)

    Returns
    -------
    sys_info : dict
        Information about an operating system and a hardware.
    """
    # import subprocess
    # import platform

    # information collected by a platform package
    sys_info = {'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor()}
    if output_node_info:
        sys_info['node'] = platform.node()

    # obtain hardware information
    try:
        with open('/dev/null', 'w') as DEVNULL:
            if platform.system() == 'Darwin':
                process_pipe = subprocess.Popen(
                    ['/usr/sbin/system_profiler', '-detailLevel',
                     'mini', 'SPHardwareDataType'],
                    stdout=subprocess.PIPE, stderr=DEVNULL)
                hard_info, _ = process_pipe.communicate()
                hard_info = hard_info.decode('utf-8').split('\n')[4:-2]
                hard_info = [i.lstrip(' ') for i in hard_info]
            elif platform.system() == 'FreeBSD':
                process_pipe = subprocess.Popen(['/sbin/sysctl', 'hw'],
                                                stdout=subprocess.PIPE,
                                                stderr=DEVNULL)
                hard_info, _ = process_pipe.communicate()
                hard_info = hard_info.decode('utf-8').split('\n')
            elif platform.system() == 'Linux':
                process_pipe = subprocess.Popen(
                    ['/bin/cat', '/proc/cpuinfo'],
                    stdout=subprocess.PIPE, stderr=DEVNULL)
                hard_info, _ = process_pipe.communicate()
                hard_info = hard_info.decode('utf-8').split('\n')
            else:
                hard_info = []
    except FileNotFoundError as e:
        print("Not linux")
        hard_info = []
    sys_info['hardware'] = hard_info

    return sys_info


def get_version_info():
    """
    Get version numbers of a Python interpreter and packages.  

    Returns
    -------
    version_info : dict
        Version numbers of a Python interpreter and packages. 
    """
    # import platform
    # import numpy as np
    # import scipy as sp
    # import sklearn

    version_info = {
        'sklearn': sklearn.__version__,
        'numpy': np.__version__, 'scipy': sp.__version__,
        'python_compiler': platform.python_compiler(),
        'python_implementation': platform.python_implementation(),
        'python': platform.python_version()}

    return version_info


# =============================================================================
# Classes
# =============================================================================

# =============================================================================
# Module initialization
# =============================================================================

# init logging system
logger = logging.getLogger('kamrecsys')
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

# =============================================================================
# Test routine
# =============================================================================


def _test():
    """ test function for this module
    """

    # perform doctest
    import sys
    import doctest

    doctest.testmod()

    sys.exit(0)


# Check if this is call as command script

if __name__ == '__main__':
    _test()
