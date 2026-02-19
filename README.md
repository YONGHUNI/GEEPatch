# GEEPatch: Spatially Aligned Satellite Data Pipeline

**GEEPatch** is a specialized Python toolkit designed to construct deep learning-ready datasets from Google Earth Engine (GEE). It enforces a strict, deterministic tiling system based on the Web Mercator grid (EPSG:3857), ensuring pixel-perfect alignment across multi-temporal satellite imagery.

This tool is optimized for high-throughput data acquisition in research environments, enabling the creation of large-scale datasets for tasks such as land-cover classification, change detection, and time-series analysis.

---

## 1. Core Rationale

Standard satellite imagery export methods (e.g., `getThumbURL` or `Export.image`) often introduce sub-pixel misalignment due to inconsistent grid anchoring or dynamic resampling heuristics. For computer vision models, this spatial noise significantly degrades performance.

**GEEPatch** addresses these challenges through three key technical decisions:

### 1.1. Deterministic Tiling (EPSG:3857)

We utilize the Web Mercator projection to create a fixed global grid system.

* **Grid Locking:** Every 256x256 patch is snapped to a pre-defined mathematical grid.
* **Temporal Consistency:** A pixel at $(x, y)$ in a 2022 image corresponds exactly to the same physical location in a 2025 image, eliminating the need for post-hoc image registration.

### 1.2. Adaptive Resolution Strategy (Sensor-Dependent)

Unlike tools hardcoded for specific sensors, `GEEPatch` requires the user to define a **Zoom Level (ZL)** that best approximates the native Ground Sampling Distance (GSD) of the target satellite. Selecting the appropriate ZL is critical to avoid under-sampling (information loss) or over-sampling (redundancy).

The ground resolution ($S$) in the Web Mercator projection is determined by:


$$S \approx \frac{40,075,016 \cdot \cos(\text{latitude})}{2^{\text{zoom} + 8}} \quad (\text{meters/pixel})$$

Users should calculate the ZL where ($S$) is closest to, but not significantly larger than, the sensor's native resolution.

| Target Sensor | Native Resolution | Recommended Zoom Level | Approx. Resolution (Equator) |
| --- | --- | --- | --- |
| **Sentinel-2** | 10m | **14** | ~9.55m |
| **Landsat 8/9** | 30m | **12** or **13** | ~38.2m / ~19.1m |
| **PlanetScope** | ~3m | **15** or **16** | ~4.77m / ~2.38m |

> **Note:** For the Sentinel-2 example in this documentation, we utilize **Zoom Level 14**, as its ~9.55m resolution closely aligns with the sensor's 10m bands.


### 1.3. Deep Learning Compatibility (8-bit PNG)

Instead of raw floating-point GeoTIFFs, data is exported as **8-bit PNGs**.

* **Normalization:** Radiometric values are linearly scaled (e.g., 0-3000 reflectance mapped to 0-255) to match standard vision backbones (ResNet, ViT).
* **Efficiency:** PNG's lossless DEFLATE compression significantly reduces I/O bottlenecks on High-Performance Computing (HPC) file systems without compromising data quality.

---

## 2. Architecture: Two-Stage Concurrent Pipeline

To maximize throughput while strictly adhering to Google Earth Engine's (GEE) API limits and local memory constraints, the `GEEPatch` core engine employs a highly optimized **Two-Stage Concurrent Processing Architecture**.

```mermaid
graph TD
    subgraph 0. Input Configuration
        A[User ROI & GEE Image] -->|geometry.py| B(Calculate EPSG:3857 Grid)
    end

    subgraph 1. Stage 1: URL Generation
        B --> C{ThreadPoolExecutor<br/>max_url_workers=16}
        C -->|tenacity @retry<br/>Exponential Backoff| D((GEE API))
        D -->|Return| E[Signed Download URLs]
    end

    subgraph 2. Stage 2: Data Streaming
        E --> F{ThreadPoolExecutor<br/>max_dl_workers}
        F -->|auth.py<br/>Connection Pooling| G((GEE Servers))
        G -->|Stream Raw .npy| H[processor.py<br/>In-Memory RAM]
    end

    subgraph 3. On-the-Fly Processing
        H -->|1. Validate Dimensions| I(Radiometric Normalization<br/>Min-Max Clip)
        I -->|2. Encode| J(8-bit PNG Converter)
    end

    subgraph 4. Output Storage
        J --> K[{XXXX}_{YYYY}_{ZL}_idx{i}.png]
        B -.->|Geometry Records| L[grid_metadata_z14.gpkg]
    end

    classDef stage fill:#f9f9f9,stroke:#333,stroke-width:2px;
    class 1.,2.,3.,4. stage;
```


### 2.1. Stage 1: Concurrent URL Generation & Exponential Backoff
* **Mechanism:** The engine first calculates the exact Web Mercator grid coordinates for the target Region of Interest (ROI). It then utilizes a `ThreadPoolExecutor` (configured via `max_url_workers`, default: 16) to concurrently request signed download URLs from the GEE API.
* **Resilience (`@retry`):** GEE tightly throttles concurrent API computations. To prevent the pipeline from crashing due to `429 Too Many Requests` or transient server errors, the URL generation method is wrapped with a `@retry` decorator (via the `tenacity` library). It employs an **exponential backoff** strategy, meaning the thread will safely pause and automatically retry the request with increasing delays until successful, ensuring 100% fault tolerance.

### 2.2. Stage 2: Concurrent Data Streaming & Processing
* **Mechanism:** Once the signed URLs are secured, a second `ThreadPoolExecutor` (configured via `max_dl_workers`) is dispatched to handle the heavy data transfer.
* **In-Memory Processing:** Instead of saving intermediate GeoTIFFs, the raw NPY data streams are downloaded directly into RAM. The data is immediately validated, radiometrically normalized, and encoded into lossless 8-bit PNGs by the processor.
* **Benefit:** This strictly avoids local I/O bottlenecks and drastically reduces the storage footprint, which is a critical advantage when operating on shared high-performance computing (HPC) file systems.


---

## 3. Installation

This toolkit relies on robust geospatial libraries. We highly recommend using `conda` (or `mamba`) via the `conda-forge` channel to gracefully handle complex C-level dependencies like GDAL.

### Option A: Conda/Mamba Environment (Recommended)

An `environment.yml` file is provided in the repository root.

```bash
# 1. Create the environment
conda env create -f environment.yml

# 2. Activate the environment
conda activate patch_fetcher

```

*(Note: The `environment.yml` specifies Python 3.11 and includes tools like `geemap` and `jupyterlab` for pipeline development.)*

### Option B: Pip (Lightweight Integration)

If you are integrating `GEEPatch` into an existing environment, you can install the core dependencies via pip.

> **Warning:** Installing `geopandas` via pip may fail if system-level GDAL binaries are not pre-installed on your OS.

```bash
pip install earthengine-api geopandas numpy pandas pillow tqdm tenacity requests

```

---

### 4. Usage

The `gee_downloader` package provides the `GEEPatch` orchestrator for automated data extraction. 

For comprehensive, large-scale workflows (including hybrid sequential/parallel processing for time-series data), please refer to the detailed Jupyter notebooks in the [`examples/`](./examples) directory:
* `examples/01_basic_usage/01_basic_usage.ipynb`
* `examples/02_steel_mine/02_steel_mine.ipynb`

### Quick Start: Single Scene Extraction
This minimal example demonstrates how to extract a single Sentinel-2 scene into Web Mercator-aligned tiles.

```python
import ee
from gee_downloader.core import GEEPatch

# 1. Initialize API and Downloader
ee.Initialize()
downloader = GEEPatch()

# 2. Define Parameters and Region of Interest (ROI)
ZOOM_LEVEL = 14
BANDS = ['B4', 'B3', 'B2']
VIS_PARAMS = {'min': [0, 0, 0], 'max': [3000, 3000, 3000]}

# Example ROI: Bochum, Germany
roi = ee.Geometry.Rectangle([7.20, 51.45, 7.25, 51.50]) 

# Define a specific Sentinel-2 acquisition (Scene)
image_id = "COPERNICUS/S2_SR_HARMONIZED/20230815T103629_20230815T104523_T32ULC"
image = ee.Image(image_id).select(BANDS)

# 3. Execute Downloader
downloader.download_as_wmts_tiles(
    image=image,
    roi=roi,
    output_dir='./dataset_example',
    zoom=ZOOM_LEVEL,
    bands=BANDS,
    vis_params=VIS_PARAMS,
    filename_prefix="demo" 
)

```

---

## 5. Output Structure & Metadata

By default, the `GEEPatch` downloader exports spatial patches into a flat directory specified by the user. Alongside the image patches, it automatically generates a vector metadata file for seamless GIS integration.

### Directory Hierarchy
For a single execution (e.g., as demonstrated in the Quick Start), the output folder contains both the 8-bit PNG images and a unified GeoPackage (`.gpkg`) file:

```text
/dataset_example/
├── demo_8515_5447_14_idx0.png
├── demo_8516_5447_14_idx1.png
├── demo_8515_5448_14_idx2.png
├── ...
└── grid_metadata_z14.gpkg        # Spatial footprints and attributes (Zoom 14)

```

### Spatial Metadata (GeoPackage)

To bridge the gap between computer vision and spatial analysis, the pipeline automatically compiles a **GeoPackage (`.gpkg`)** file.

* **Format:** `grid_metadata_z{zoom}.gpkg`
* **Contents:** Contains the exact spatial bounding boxes (Polygons) for every successfully downloaded patch.
* **Attributes:** Stores critical metadata including the Web Mercator tile coordinates (`xtile`, `ytile`, `zoom`), the corresponding image file name, and the acquisition date.


### Naming Convention (Images)

The final image filename is dynamically constructed using the Web Mercator tile coordinates to ensure strict spatial traceability, followed by a sequential batch index.

> **Format:** `[{filename_prefix}_]{xtile}_{ytile}_{zoom}_idx{PatchIndex}.png`

* **`xtile`, `ytile`, `zoom` (Auto-Generated):** The fundamental Web Mercator grid coordinates for the specific 256x256 patch.
* **`PatchIndex` (Auto-Generated):** A unique, zero-based sequence integer (`0`, `1`, `2`, ...) for the tiles processed within the current bounding box.
* **`filename_prefix` (Optional Parameter):**
* **Case 1: Custom Prefix Provided**
The user-defined string is prepended to the base name.
> Example: `filename_prefix="demo"` yields `demo_8515_5447_14_idx0.png`.


* **Case 2: Omitted (Default Base Name)**
If left blank, the system relies strictly on the mathematical grid coordinates.
> Example: `8515_5447_14_idx0.png`





> **Note for Large-Scale Datasets:**
> For multi-temporal or multi-region datasets, users can build hierarchical directory structures (e.g., grouping by region or meteorological season) by wrapping the downloader in custom Python loops. See the `examples/02_steel_mine/` directory for an advanced implementation.


---

## 6. Project Modules

The codebase is modularized for maintainability and clear separation of concerns:

* `core.py`: Main orchestrator containing the `download_as_wmts_tiles` logic, API calling, and thread management.
* `geometry.py`: Mathematical functions for Coordinate Reference System (CRS) transformations and affine matrix calculations.
* `processor.py`: Handles raw NPY data stream decoding, radiometric validation, and PNG encoding.
* `auth.py`: Manages Google Earth Engine initialization and session authentication.
