import unittest
import sys
import os
import tempfile
import numpy as np
import io
from PIL import Image
from unittest.mock import MagicMock, patch

# Resolve the absolute path of the current file (tests/test_geepatch.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Move one level up to get the project root (patch_fetcher)
project_root = os.path.dirname(current_dir)

# Insert the project root at the beginning of sys.path to prioritize local modules
sys.path.insert(0, project_root)

# Internal module imports
import gee_downloader.geometry as geometry
import gee_downloader.processor as processor
from gee_downloader.core import GEEPatch

class TestGeometry(unittest.TestCase):
    """
    [Geometry Module Test]
    Tests mathematical precision of tile coordinates and affine transforms.
    """
    
    def test_deg2num_consistency(self):
        """Verify that coordinate conversion is reversible (Lat/Lon <-> Tile X/Y)."""
        # Test Case: Bochum coordinates
        lat, lon = 51.48, 7.21
        zoom = 14
        xtile, ytile = geometry.deg2num(lat, lon, zoom)
        
        # Reverse check
        lat_check, lon_check = geometry.num2deg(xtile, ytile, zoom)
        
        # Check if reversed coordinates are within 1 tile tolerance
        # (Note: Tile conversion is discrete, introducing inherent quantization error)
        self.assertTrue(abs(lat - lat_check) < 0.1, "Latitude conversion failed")
        self.assertTrue(abs(lon - lon_check) < 0.1, "Longitude conversion failed")
        
    def test_affine_transform_structure(self):
        """Verify the affine transform matrix structure and bounds logic."""
        xtile, ytile, zoom = 8520, 5448, 14
        transform, bounds = geometry.get_affine_transform(xtile, ytile, zoom)
        
        # 1. Transform must be a list/tuple of 6 floats
        self.assertEqual(len(transform), 6)
        # 2. Scale X must be positive, Scale Y negative (Standard for Web Mercator)
        self.assertGreater(transform[0], 0, "Scale X must be positive")
        self.assertLess(transform[4], 0, "Scale Y must be negative")
        # 3. Bounds sanity check (MinX < MaxX, MinY < MaxY)
        self.assertLess(bounds[0], bounds[2])
        self.assertLess(bounds[1], bounds[3])

class TestProcessor(unittest.TestCase):
    """
    [Processor Module Test]
    Tests data validation, normalization logic, and PNG encoding.
    """
    
    def setUp(self):
        """Initialize a temporary directory for safe, isolated I/O testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = self.temp_dir.name
        
    def tearDown(self):
        """Automatically clean up the temporary directory regardless of test success/failure."""
        self.temp_dir.cleanup()

    def test_validate_vis_params_broadcasting(self):
        """Test if scalar min/max values are correctly broadcasted to all bands."""
        bands = ['B4', 'B3', 'B2']
        vis = {'min': 0, 'max': 3000}
        vmin, vmax = processor.validate_vis_params(bands, vis)
        
        # Should broadcast to shape (1, 1, 3)
        self.assertEqual(vmin.shape, (1, 1, 3))
        self.assertTrue(np.all(vmin == 0))
        self.assertTrue(np.all(vmax == 3000))

    def test_validate_vis_params_invalid_range(self):
        """Test if the validator catches invalid ranges (Min > Max)."""
        bands = ['B4']
        vis = {'min': 4000, 'max': 3000} # Invalid: Min > Max
        with self.assertRaises(ValueError):
            processor.validate_vis_params(bands, vis)

    def test_process_npy_to_png_pixel_values(self):
        """
        Verify the exact pixel value mapping from Raw NPY to PNG.
        Simulates the full processing pipeline without requiring network access.
        """
        # 1. Create a dummy structured NPY buffer
        dtype = [('B4', 'f4'), ('B3', 'f4'), ('B2', 'f4')]
        arr = np.zeros((256, 256), dtype=dtype)
        
        # Set specific test values
        arr['B4'] = 1500 # Midpoint (should be exactly mapped to ~127.5)
        arr['B3'] = 0    # Min (should be 0)
        arr['B2'] = 3000 # Max (should be 255)
        
        buffer = io.BytesIO()
        np.save(buffer, arr)
        buffer.seek(0)
        
        output_path = os.path.join(self.test_dir, "test_pixel.png")
        bands = ['B4', 'B3', 'B2']
        # Set visualization range 0-3000
        vmin, vmax = np.array([[[0, 0, 0]]]), np.array([[[3000, 3000, 3000]]])
        
        # 2. Run Processor
        processor.process_npy_to_png(buffer, output_path, bands, vmin, vmax)
        
        # 3. Check Results
        self.assertTrue(os.path.exists(output_path), "PNG file was not created")
        
        with Image.open(output_path) as img:
            self.assertEqual(img.size, (256, 256))
            self.assertEqual(img.mode, 'RGB')
            
            # Sample a pixel to verify normalization math
            pixel = img.getpixel((128, 128))
            
            # B4: 1500 / 3000 * 255 = 127.5 -> rounded to 127 or 128
            self.assertIn(pixel[0], (127, 128), f"Pixel Value B4 mismatch: {pixel[0]}")
            self.assertEqual(pixel[1], 0, "Pixel Value B3 mismatch")
            self.assertEqual(pixel[2], 255, "Pixel Value B2 mismatch")

class TestCore(unittest.TestCase):
    """
    [Core Module Test]
    Tests the main GEEPatch class orchestration using Mocks.
    """
    
    @patch('gee_downloader.auth.initialize_gee')
    def test_init_workers(self, mock_init):
        """Test initialization and worker scaling logic."""
        # Force 8 workers
        downloader = GEEPatch(max_dl_workers=8)
        self.assertEqual(downloader.MAX_DL_WORKERS, 8)
        mock_init.assert_called_once()
        
    @patch('gee_downloader.auth.initialize_gee')
    def test_generate_signed_url_call(self, mock_init):
        """Test if _generate_signed_url correctly calls the GEE API."""
        downloader = GEEPatch()
        
        # Mock an ee.Image object
        mock_image = MagicMock()
        mock_image.getDownloadURL.return_value = "http://mock.url/download"
        
        dummy_params = {'region': 'mock'}
        url = downloader._generate_signed_url(mock_image, dummy_params)
        
        self.assertEqual(url, "http://mock.url/download")
        mock_image.getDownloadURL.assert_called_with(dummy_params)

    @patch('gee_downloader.auth.initialize_gee')
    def test_download_as_wmts_tiles_format_validation(self, mock_init):
        """
        Verify that the orchestrator rejects unsupported output formats 
        before initiating any heavy API calls or geometry processing.
        """
        downloader = GEEPatch()
        
        # Provide dummy objects for arguments
        dummy_image = MagicMock()
        dummy_roi = MagicMock()
        
        # Expect a ValueError to be raised for an unsupported format like 'jpg'
        with self.assertRaisesRegex(ValueError, "Unsupported output format"):
            downloader.download_as_wmts_tiles(
                image=dummy_image,
                roi=dummy_roi,
                output_dir="./dummy_dir",
                zoom=14,
                bands=['B4'],
                out_format='jpg' # Invalid format trigger
            )

    @patch('gee_downloader.auth.initialize_gee')
    def test_download_raw_tiff_bypass(self, mock_init):
        """
        Test the GEO_TIFF download method.
        Verifies that binary chunks are streamed directly to disk without loading 
        the full image into RAM or passing through the processor.
        """
        downloader = GEEPatch()
        
        downloader.session.get = MagicMock()
        
        # 1. Mock the requests.Session response and its chunk stream
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"mock_", b"tiff_", b"data"]
        
        # Mock the context manager behavior of the session
        downloader.session.get.return_value.__enter__.return_value = mock_response
        
        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = os.path.join(temp_dir, "test_output.tif")
            
            # 2. Execute the raw bypass download
            downloader._download_raw_tiff("http://mock.url/tiff", out_path)
            
            # 3. Verify that the file was written to disk and chunks were concatenated
            self.assertTrue(os.path.exists(out_path), "TIFF file was not saved to disk")
            
            with open(out_path, 'rb') as f:
                content = f.read()
            
            self.assertEqual(content, b"mock_tiff_data", "File content mismatch")
            mock_response.raise_for_status.assert_called_once()

if __name__ == '__main__':
    unittest.main()