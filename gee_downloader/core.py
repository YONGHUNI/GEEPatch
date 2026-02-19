"""
Core GEEPatch Orchestrator.

This module combines authentication, geometry calculations, and image processing
to execute a robust satellite imagery download pipeline. It supports high-throughput
concurrency using a batch processing strategy (URL generation -> File Download).
"""

import os
import io
import ee
import geopandas as gpd
from shapely.geometry import box, shape
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests
from typing import Optional, List, Dict, Union

# Import internal modules
from . import auth, geometry, processor

class GEEPatch:
    """
    Main controller for downloading strictly aligned satellite imagery patches.
    
    Attributes:
        MAX_URL_WORKERS (int): Number of threads for GEE API calls (URL generation).
        MAX_DL_WORKERS (int): Number of threads for file downloading and processing.
        MAX_TILES_LIMIT (int): Safety limit to prevent accidental bulk downloads.
        session (requests.Session): Persistent HTTP session with retry logic.
    """

    def __init__(self, max_url_workers: int = 16, max_dl_workers: Optional[int] = None):
        """
        Initializes the GEEPatch downloader.

        Args:
            max_url_workers (int): Thread count for GEE API requests. 
                                   Keep moderate (e.g., 16) to avoid rate limits.
            max_dl_workers (Optional[int]): Thread count for I/O bound downloads.
                                            Defaults to 4x CPU count (clamped at 64).
        """
        self.MAX_URL_WORKERS = max_url_workers
        
        # Heuristic: Optimize for I/O bound tasks
        if max_dl_workers is None:
            cpu_count = os.cpu_count() or 4
            self.MAX_DL_WORKERS = max(4, min(cpu_count * 4, 64))
        else:
            self.MAX_DL_WORKERS = max_dl_workers

        self.MAX_TILES_LIMIT = 5000
        
        # Initialize Google Earth Engine and HTTP session
        auth.initialize_gee()
        self.session = auth.get_session(self.MAX_DL_WORKERS)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=20),
           retry=retry_if_exception_type(ee.EEException))
    def _generate_signed_url(self, image: ee.Image, params: Dict) -> str:
        """
        Retrieves a signed download URL from Google Earth Engine.
        Includes retry logic for transient API errors.
        """
        return image.getDownloadURL(params)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10),
           retry=retry_if_exception_type(requests.exceptions.RequestException))
    def _download_and_process(self, url: str, output_path: str, bands: List[str], vmin, vmax):
        """
        Downloads data from the provided URL and processes it into a PNG image.
        
        This method streams the NPY data into memory, converts it using the processor module,
        and saves the result to disk.
        """
        buffer = io.BytesIO()
        try:
            with self.session.get(url, stream=True, timeout=(30, 60)) as response:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk: buffer.write(chunk)
            processor.process_npy_to_png(buffer, output_path, bands, vmin, vmax)
        finally:
            buffer.close()

    def download_as_wmts_tiles(
        self, 
        image: ee.Image, 
        roi: Union[ee.Geometry, ee.FeatureCollection, gpd.GeoDataFrame, gpd.GeoSeries], 
        output_dir: str, 
        zoom: int, 
        bands: List[str], 
        filename_prefix: Optional[str] = None,
        vis_params: Optional[Dict] = None,
        show_progress: bool = True
    ) -> Union[gpd.GeoDataFrame, None]:
        """
        Executes the tiled download pipeline for a specific satellite image within an ROI.

        The pipeline operates in two phases to maximize throughput:
        1. Grid Calculation: Determines the Web Mercator tiles intersecting the ROI.
        2. Phase 1 (URL Generation): Fetches signed URLs concurrently (I/O bound).
        3. Phase 2 (Download & Process): Downloads and converts data concurrently.

        Args:
            image (ee.Image): The GEE image object to download.
            roi (Union[...]): The Region of Interest (Geometry or GeoDataFrame).
            output_dir (str): Local directory to save PNGs and metadata.
            zoom (int): Web Mercator zoom level (determines resolution).
            bands (List[str]): List of bands to visualize (e.g., ['B4', 'B3', 'B2']).
            filename_prefix (Optional[str]): Custom suffix for filenames. Format: {x}_{y}_{z}_{prefix}.png.
            vis_params (Optional[Dict]): Visualization parameters. Default: {'min': 0, 'max': 3000}.
            show_progress (bool): If True, displays a tqdm progress bar for the download phase. 
                                  Set to False to reduce console clutter in nested loops.

        Returns:
            Union[gpd.GeoDataFrame, None]: A GeoDataFrame containing tile metadata if successful, else None.
        """
        
        # Ensure output directory exists
        if not os.path.exists(output_dir): 
            os.makedirs(output_dir, exist_ok=True)
            
        # Default visualization parameters (Sentinel-2 L2A optimized)
        vis_params = vis_params or {"min": 0, "max": 3000}
        
        # 1. Validation of Visualization Parameters
        try:
            vmin_arr, vmax_arr = processor.validate_vis_params(bands, vis_params)
        except ValueError as e:
            print(f"Configuration Error: {e}")
            return None

        # 2. ROI Parsing and Coordinate Transformation
        roi_shapely = None
        try:
            if isinstance(roi, (gpd.GeoDataFrame, gpd.GeoSeries)):
                # Ensure CRS is present before transformation
                if roi.crs is None:
                    raise ValueError("GeoPandas object must have a CRS defined.")
                roi_shapely = roi.to_crs(epsg=4326).geometry.unary_union
            elif isinstance(roi, (ee.Geometry, ee.Feature, ee.FeatureCollection)):
                # Convert GEE object to Shapely via GeoJSON
                roi_shapely = shape(geometry.extract_geometry(roi.getInfo()))
            else:
                raise TypeError("ROI must be an ee.Geometry or GeoPandas object.")
        except Exception as e:
            print(f"Error parsing geometry: {e}")
            return None

        # 3. Grid Calculation (Web Mercator Tile Indices)
        bounds = roi_shapely.bounds
        min_x, min_y = geometry.deg2num(bounds[3], bounds[0], zoom)
        max_x, max_y = geometry.deg2num(bounds[1], bounds[2], zoom)
        
        tasks = []
        grid_records = {} 
        tile_index = 0

        # 4. Task Creation Loop
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                # Calculate tile boundaries in Lat/Lon
                n, w = geometry.num2deg(x, y, zoom)
                s, e = geometry.num2deg(x + 1, y + 1, zoom)
                
                # Perform strict intersection check
                if not roi_shapely.intersects(box(w, s, e, n)):
                    continue

                # Construct filename
                if filename_prefix:
                    filename = f"{x}_{y}_{zoom}_{filename_prefix}.png"
                else:
                    filename = f"{x}_{y}_{zoom}_idx{tile_index}.png"
                    
                filepath = os.path.join(output_dir, filename)
                
                # Get Affine Transform for GEE export
                transform, (mx_min, my_min, mx_max, my_max) = geometry.get_affine_transform(x, y, zoom)
                
                # Deduplication: Skip if file already exists
                if not os.path.exists(filepath):
                    tasks.append({
                        'image': image, 
                        'region': ee.Geometry.Rectangle([mx_min, my_min, mx_max, my_max], proj='EPSG:3857', geodesic=False),
                        'transform': transform,
                        'filepath': filepath,
                        'id': filename,
                        'url': None
                    })
                    status = 'pending'
                else:
                    status = 'exists_skipped'

                # Create Metadata Record (including empty label column)
                grid_records[filename] = {
                    'tile_index': tile_index,
                    'x': x, 'y': y, 'zoom': zoom,
                    'label': None,  # Initialize as None (NULL in GPKG) for manual labeling
                    'filename': filename,
                    'geometry': box(mx_min, my_min, mx_max, my_max),
                    'status': status,
                    'error': None
                }
                tile_index += 1

        # Return early if no tiles intersect or all exist
        if not tasks:
            if grid_records:
                 gdf = gpd.GeoDataFrame(list(grid_records.values()), crs="EPSG:3857")
                 return gdf
            return None

        # 5. Phase 1: URL Generation (Batch Mode)
        # Fetch URLs in parallel using MAX_URL_WORKERS
        download_ready_tasks = []
        with ThreadPoolExecutor(max_workers=self.MAX_URL_WORKERS) as executor:
            future_to_task = {}
            for task in tasks:
                # Clip image to tile geometry to avoid edge artifacts
                img_proc = task['image'].resample('bicubic').clip(task['region'])
                params = {
                    'crs': 'EPSG:3857', 
                    'crs_transform': task['transform'], 
                    'dimensions': '256x256', 
                    'format': 'NPY'
                }
                future_to_task[executor.submit(self._generate_signed_url, img_proc, params)] = task

            # Collect results (No progress bar here for cleaner UI)
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    task['url'] = future.result()
                    download_ready_tasks.append(task)
                except Exception:
                    grid_records[task['id']]['status'] = 'failed_url'

        # 6. Phase 2: Download and Processing
        # Execute heavy I/O tasks using MAX_DL_WORKERS
        if download_ready_tasks:
            with ThreadPoolExecutor(max_workers=self.MAX_DL_WORKERS) as executor:
                future_to_id = {}
                for task in download_ready_tasks:
                    future = executor.submit(
                        self._download_and_process, 
                        task['url'], 
                        task['filepath'], 
                        bands, 
                        vmin_arr, 
                        vmax_arr
                    )
                    future_to_id[future] = task['id']
                
                # Optional Progress Bar controlled by 'show_progress'
                for future in tqdm(as_completed(future_to_id), 
                                  total=len(future_to_id), 
                                  desc="Processing Patches", 
                                  leave=False, 
                                  disable=not show_progress): 
                    tile_id = future_to_id[future]
                    try:
                        future.result()
                        grid_records[tile_id]['status'] = 'success'
                    except Exception as e:
                        grid_records[tile_id]['status'] = 'failed_process'
                        grid_records[tile_id]['error'] = str(e)

        # 7. Metadata Export
        if grid_records:
            try:
                gdf = gpd.GeoDataFrame(list(grid_records.values()), crs="EPSG:3857")
                gpkg_path = os.path.join(output_dir, f"grid_metadata_z{zoom}.gpkg")
                gdf.to_file(gpkg_path, driver="GPKG")
                return gdf
            except Exception:
                pass

        return None