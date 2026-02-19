"""
Geometry and Coordinate Math for GEEPatch.
Focuses on Web Mercator (EPSG:3857) tile calculations and geometry extraction.
"""

import math
from typing import Tuple, List, Any

def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> Tuple[int, int]:
    """
    Converts WGS84 coordinates (lat/lon) to Web Mercator tile X/Y indices.
    """
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile

def num2deg(xtile: int, ytile: int, zoom: int) -> Tuple[float, float]:
    """
    Converts Web Mercator tile X/Y indices to the NW corner lat/lon.
    """
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg

def get_affine_transform(xtile: int, ytile: int, zoom: int) -> Tuple[List[float], Tuple[float, float, float, float]]:
    """
    Calculates the Affine Transform Matrix and Bounds for a specific tile.
    
    This ensures the requested grid aligns mathematically with EPSG:3857.
    
    Returns:
        transform (List[float]): [ScaleX, ShearX, TransX, ShearY, ScaleY, TransY] for GEE.
        bounds (Tuple): (min_x, min_y, max_x, max_y) in EPSG:3857 coordinates.
    """
    earth_circ = 2 * math.pi * 6378137.0
    origin_shift = earth_circ / 2.0 
    resolution = earth_circ / (256 * (2**zoom))

    # Calculate origin in meters (EPSG:3857)
    offset_x = (xtile * 256) * resolution - origin_shift
    offset_y = origin_shift - (ytile * 256) * resolution

    min_x = offset_x
    max_y = offset_y
    max_x = offset_x + (256 * resolution)
    min_y = offset_y - (256 * resolution)

    # Matrix for GEE getDownloadURL
    transform = [resolution, 0, offset_x, 0, -resolution, offset_y]
    bounds = (min_x, min_y, max_x, max_y)
    
    # Runtime integrity check (Precision Guard)
    if abs((max_x - min_x) - (256 * resolution)) > 1e-9:
        raise ValueError("Math Error: Resolution/Bounds mismatch in geometry calculation.")
        
    return transform, bounds

def extract_geometry(roi_obj: Any) -> Any:
    """
    Safely extracts the geometry object from various GEE inputs 
    (Geometry, Feature, FeatureCollection).
    """
    if 'type' not in roi_obj:
        raise ValueError("Invalid ROI object structure.")
    
    if roi_obj['type'] in ['Geometry', 'Polygon', 'MultiPolygon', 'Rectangle']:
        return roi_obj
    elif roi_obj['type'] == 'Feature':
        return roi_obj['geometry']
    elif roi_obj['type'] == 'FeatureCollection':
        # Warning: Using the first feature of the collection
        return roi_obj['features'][0]['geometry']
    else:
        raise ValueError(f"Unsupported geometry type: {roi_obj['type']}")