"""
Authentication and Session Management for GEEPatch.
"""

import ee
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def initialize_gee():
    """
    Initializes Google Earth Engine.
    Attempts explicit authentication if the default initialization fails.
    """
    try:
        ee.Initialize()
    except Exception:
        print("Standard initialization failed. Attempting explicit authentication...")
        try:
            ee.Authenticate()
            ee.Initialize()
        except Exception as e:
            raise RuntimeError(f"Critical: GEE Authentication failed. {e}")
    print("Google Earth Engine initialized successfully.")

def get_session(max_workers: int) -> requests.Session:
    """
    Creates a requests Session with retry logic for robust downloads.
    
    Args:
        max_workers (int): Pool size for concurrent connections.
    """
    session = requests.Session()
    retries = Retry(
        total=5, 
        backoff_factor=1, 
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"], 
        respect_retry_after_header=True
    )
    adapter = HTTPAdapter(
        pool_connections=max_workers, 
        pool_maxsize=max_workers, 
        max_retries=retries
    )
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session