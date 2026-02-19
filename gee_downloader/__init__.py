"""
GEEPatch: Google Earth Engine Image Patch Extractor.

This package provides a robust pipeline to download satellite imagery 
strictly aligned to the Web Mercator (EPSG:3857) grid for deep learning applications.
"""

import os
import sys

# =========================================================
# [Critical Fix] Auto-configure PROJ library path
# Prevents "proj.db not found" errors in GeoPandas/GDAL
# =========================================================
try:
    import pyproj
    # Retrieve the actual data directory from pyproj
    proj_lib_path = pyproj.datadir.get_data_dir()
    os.environ['PROJ_LIB'] = proj_lib_path
except ImportError:
    # Fallback: Estimate Conda path if pyproj is missing
    conda_prefix = os.path.dirname(os.path.dirname(sys.executable))
    estimated_path = os.path.join(conda_prefix, 'share', 'proj')
    if os.path.exists(estimated_path):
        os.environ['PROJ_LIB'] = estimated_path

# Expose the main class for easy access
from .core import GEEPatch