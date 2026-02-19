"""
Data Processing Module for GEEPatch.
Handles NPY streaming validation, normalization, and atomic image encoding.
"""

import io
import os
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple

def validate_vis_params(bands: List[str], vis_params: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validates visualization parameters (min/max).
    Ensures they match the band count and that min < max.
    """
    vmin = vis_params.get('min', 0)
    vmax = vis_params.get('max', 3000)

    def to_array(val, name):
        if isinstance(val, (int, float)):
            return np.full((1, 1, len(bands)), val, dtype=float)
        elif isinstance(val, (list, tuple, np.ndarray)):
            if len(val) != len(bands):
                raise ValueError(f"{name} length must match band count ({len(bands)}).")
            return np.array(val, dtype=float).reshape(1, 1, -1)
        raise ValueError(f"{name} must be a numeric value or a list.")

    vmin_arr = to_array(vmin, 'min')
    vmax_arr = to_array(vmax, 'max')

    if np.any(vmin_arr >= vmax_arr):
        raise ValueError("Normalization Error: 'min' must be strictly less than 'max'.")

    return vmin_arr, vmax_arr

def process_npy_to_png(buffer: io.BytesIO, output_path: str, bands: List[str], vmin: np.ndarray, vmax: np.ndarray, nan_policy: str = 'zero'):
    """
    Decodes NPY buffer, applies normalization, and saves as PNG.
    
    Steps:
    1. Validate Structured Array (Band integrity).
    2. Check Dtypes (No objects/strings).
    3. Normalize (Linear Stretch) to 0-255 uint8.
    4. Atomic Save (Write tmp -> Rename).
    """
    tmp_path = None
    try:
        buffer.seek(0)
        arr = np.load(buffer, allow_pickle=True)

        # 1. Strict Band Validation
        if not arr.dtype.names:
            raise ValueError("Received Plain Array. GEE must return Structured Array to ensure band order.")

        available_bands = arr.dtype.names
        missing_bands = [b for b in bands if b not in available_bands]
        if missing_bands:
            raise ValueError(f"Missing bands: {missing_bands}. Available: {available_bands}")
        
        # 2. Safety Check
        for b in bands:
            if arr[b].dtype.kind not in ['f', 'i', 'u']:
                raise ValueError(f"Band {b} has unsafe dtype kind. Expected float/int.")

        # Stack bands strictly in requested order
        img_data = np.dstack([arr[b] for b in bands])

        # 3. Shape Validation (256x256)
        if img_data.shape[0] != 256 or img_data.shape[1] != 256:
            raise ValueError(f"Invalid tile dimensions: {img_data.shape[:2]}. Expected 256x256.")

        # 4. NaN Policy & Normalization
        if nan_policy == 'zero':
            img_data = np.nan_to_num(img_data, nan=0.0)

        denom = vmax - vmin
        denom = np.where(denom <= 1e-9, 1.0, denom) # Guard zero div

        img_data = np.clip(img_data, vmin, vmax)
        img_data = (img_data - vmin) / denom * 255.0
        img_data = img_data.astype(np.uint8)

        # 5. Determine Mode & Save
        if img_data.shape[2] == 1:
            mode = 'L'
            img_data = img_data[:, :, 0]
        elif img_data.shape[2] == 3:
            mode = 'RGB'
        elif img_data.shape[2] == 4:
            mode = 'RGBA'
        else:
            raise ValueError(f"Unsupported channel count: {img_data.shape[2]}")

        tmp_path = output_path + ".tmp.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        Image.fromarray(img_data, mode).save(tmp_path, format='PNG')
        os.replace(tmp_path, output_path)

    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except OSError: pass
        # Re-raise to be caught by the worker thread
        raise RuntimeError(f"Processing failed for {output_path}: {e}")