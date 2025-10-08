from sentinelhub import (
    SHConfig,
    DataCollection,
    SentinelHubCatalog,
    SentinelHubRequest,
    BBox,
    bbox_to_dimensions,
    CRS,
    BBoxSplitter
)
import os
import tarfile
import json
import shutil
from pathlib import Path
import time
import datetime
from contextlib import contextmanager

import os
project_root = '/home/envitwin/Desktop/venvs/EnviTwin/Operational_LULC'


@contextmanager
def timer(name):
    """A context manager to time a block of code."""
    print(f"\nStarting: {name}...")
    start_time = time.time()
    yield
    end_time = time.time()
    print(f"Finished: {name}. Duration: {end_time - start_time:.2f} seconds.")

def clean_date(dt_str: str) -> str:
    """Normalize Sentinel datetime string to keep only the date (YYYY-MM-DD)."""
    return dt_str.split("T")[0]

def organize_sentinel_downloads(base_download_dir: str, metadata: dict, base_output_dir: str = project_root + "/data"):
    """Organize SentinelHub downloads into a structured directory including a unique area ID."""
    try:
        with timer(f"Organizing {base_download_dir}"):
            base_download_dir = Path(base_download_dir)
            if not base_download_dir.exists():
                raise FileNotFoundError(f"Base download directory not found: {base_download_dir}")

            sensor = base_download_dir.parent.name
            subdirs = sorted([d for d in base_download_dir.iterdir() if d.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
            if not subdirs:
                raise FileNotFoundError(f"No product directories found in {base_download_dir}")

            product_dir = subdirs[0]
            tar_files = list(product_dir.glob("*.tar"))
            if not tar_files:
                raise FileNotFoundError(f"No .tar file found in {product_dir}")
            
            tar_file = tar_files[0]
            date_str = metadata.get("date_time")
            if not date_str:
                raise ValueError("Metadata must contain 'date_time'")
            date_str = clean_date(date_str)

            # Create a unique ID for the area based on its bounding box
            bbox_tuple = tuple(metadata.get("bbox", []))
            area_id = abs(hash(bbox_tuple))

            # Create clean output dir: ./data/SENSOR/YYYY-MM-DD/AREA_ID/
            output_dir = Path(base_output_dir) / sensor / date_str / str(area_id)
            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"Extracting {tar_file} -> {output_dir}")
            with tarfile.open(tar_file, "r") as tf:
                tf.extractall(output_dir)

            metadata_file = output_dir / "metadata.json"
            print(f"Saving metadata -> {metadata_file}")
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=4)

            print(f"Deleting original folder {product_dir}")
            shutil.rmtree(product_dir)

            return str(output_dir)
    except (FileNotFoundError, ValueError, tarfile.ReadError) as e:
        print(f"Error in organize_sentinel_downloads: {e}")
        if 'product_dir' in locals() and product_dir.exists():
            shutil.rmtree(product_dir)
            print(f"Deleted corrupted folder {product_dir}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred in organize_sentinel_downloads: {e}")
        return None

def _check_existing_data(output_dir: Path, expected_files: list) -> bool:
    """Checks if the output directory exists and contains the expected files."""
    if not output_dir.is_dir():
        return False
    if not (output_dir / "metadata.json").is_file():
        return False
    for f in expected_files:
        if not (output_dir / f).is_file():
            return False
    return True

def download_sentinel2(config, config_run, aoi_bbox, aoi_size, resolution):
    """Downloads Sentinel-2 L2A bands with retry logic and returns a status dictionary."""
    print("\n----- Sentinel-2 Download ----- ")
    try:
        with timer("Searching for S2 images"):
            catalog = SentinelHubCatalog(config=config)
            time_interval_s2 = config_run["S2_time_interval"]
            mosaickingOrder = config_run["mosaickingOrderS2"]

            search_iterator = catalog.search(
                DataCollection.SENTINEL2_L2A,
                bbox=aoi_bbox,
                time=time_interval_s2,
                fields={"include": ["id", "properties.datetime", "properties.eo:cloud_cover"], "exclude": []},
            )
            results = list(search_iterator)
            print(f"Found {len(results)} S2 results.")

            if not results:
                raise ValueError("No Sentinel-2 results found for given parameters.")

            if mosaickingOrder == "mostRecent":
                selected_result = max(results, key=lambda x: x['properties']['datetime'])
            elif mosaickingOrder == "leastCC":
                selected_result = min(results, key=lambda x: x['properties']['eo:cloud_cover'])
            else: # leastRecent
                selected_result = min(results, key=lambda x: x['properties']['datetime'])
            print(f"Selected S2 image from {clean_date(selected_result['properties']['datetime'])} with cloud cover {selected_result['properties']['eo:cloud_cover']:.2f}%")

        metadata_s2 = {
            "bbox": list(aoi_bbox),
            "crs": str(aoi_bbox.crs),
            "resolution": resolution,
            "date_time": clean_date(selected_result['properties']['datetime']),
            "cloud_cover": selected_result['properties']['eo:cloud_cover'],
            "image_id": selected_result['id']
        }

        # Calculate target output directory to check for existing data
        date_str_s2 = metadata_s2["date_time"]
        bbox_tuple = tuple(metadata_s2.get("bbox", []))
        area_id = abs(hash(bbox_tuple))
        target_output_dir = Path(project_root + "/data") / "S2" / date_str_s2 / str(area_id)

        expected_s2_files = [f"B{i:02d}.tif" for i in range(1, 13) if i not in [10]] + ["B8A.tif"]
        if _check_existing_data(target_output_dir, expected_s2_files):
            print(f"Data for S2 at {date_str_s2} for area {area_id} already exists. Skipping download.")
            return {"status": "SKIPPED", "metadata": metadata_s2, "cost": 0.0}

        evalscript_s2_bands = '''
            //VERSION=3
            function setup() {
                return {
                    input: [{ bands: ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12"] }],
                    output: [
                        {id: "B01", bands:1, sampleType: SampleType.FLOAT32},{id: "B02", bands:1, sampleType: SampleType.FLOAT32},
                        {id: "B03", bands:1, sampleType: SampleType.FLOAT32},{id: "B04", bands:1, sampleType: SampleType.FLOAT32},
                        {id: "B05", bands:1, sampleType: SampleType.FLOAT32},{id: "B06", bands:1, sampleType: SampleType.FLOAT32},
                        {id: "B07", bands:1, sampleType: SampleType.FLOAT32},{id: "B08", bands:1, sampleType: SampleType.FLOAT32},
                        {id: "B8A", bands:1, sampleType: SampleType.FLOAT32},{id: "B09", bands:1, sampleType: SampleType.FLOAT32},
                        {id: "B11", bands:1, sampleType: SampleType.FLOAT32},{id: "B12", bands:1, sampleType: SampleType.FLOAT32}
                    ]
                };
            }
            function evaluatePixel(sample) {
                return {
                    B01:[sample.B01], B02:[sample.B02], B03:[sample.B03], B04:[sample.B04],
                    B05:[sample.B05], B06:[sample.B06], B07:[sample.B07], B08:[sample.B08],
                    B8A:[sample.B8A], B09:[sample.B09], B11:[sample.B11], B12:[sample.B12]
                };
            }
        '''

        s2_data_folder = Path(project_root + f"/data/S2/{date_str_s2}")
        s2_data_folder.mkdir(parents=True, exist_ok=True)

        request_s2_bands = SentinelHubRequest(
            data_folder=str(s2_data_folder),
            evalscript=evalscript_s2_bands,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A.define_from(
                        name="s2l2a", service_url="https://sh.dataspace.copernicus.eu"
                    ),
                    time_interval=(selected_result['properties']['datetime'], selected_result['properties']['datetime']),
                )
            ],
            responses=[{"identifier": band, "format": {"type": "image/tiff"}} for band in ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12"]],
            bbox=aoi_bbox,
            size=aoi_size,
            config=config,
        )
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with timer("Downloading S2 data"):
                    download_responses = request_s2_bands.get_data(save_data=True, decode_data=False)
                    cost_str = download_responses[0].headers['x-processingunits-spent']
                    cost_pu = float(cost_str)
                    print(f"Download successful. Copernicus PU cost: {cost_pu:.2f}")
                    return {"status": "SUCCESS", "metadata": metadata_s2, "cost": cost_pu}
            except Exception as e:
                print(f"Attempt {attempt + 1} failed with error: {e}")
                if attempt < max_retries - 1:
                    print("Waiting before retrying...")
                    time.sleep(5)
                else:
                    print("All download attempts failed for Sentinel-2.")
                    return {"status": "FAILED", "error_message": str(e)}
    except Exception as e:
        return {"status": "FAILED", "error_message": str(e)}

def download_sentinel1(config, config_run, aoi_bbox, aoi_size, resolution):
    """Downloads Sentinel-1 IW bands with retry logic and returns a status dictionary."""
    print("\n----- Sentinel-1 Download ----- ")
    try:
        with timer("Searching for S1 images"):
            catalog = SentinelHubCatalog(config=config)
            time_interval_s1 = config_run["S1_time_interval"]
            mosaickingOrder = config_run["mosaickingOrderS1"]

            search_iterator = catalog.search(
                DataCollection.SENTINEL1_IW,
                bbox=aoi_bbox,
                time=time_interval_s1,
                fields={"include": ["id", "properties.datetime"], "exclude": []},
            )
            results = list(search_iterator)
            print(f"Found {len(results)} S1 results.")

            if not results:
                raise ValueError("No Sentinel-1 results found for given parameters.")

            if mosaickingOrder == "mostRecent":
                selected_result = max(results, key=lambda x: x['properties']['datetime'])
            else: # leastRecent
                selected_result = min(results, key=lambda x: x['properties']['datetime'])
            print(f"Selected S1 image from {clean_date(selected_result['properties']['datetime'])}")

        metadata_s1 = {
            "bbox": list(aoi_bbox),
            "crs": str(aoi_bbox.crs),
            "resolution": resolution,
            "date_time": clean_date(selected_result['properties']['datetime']),
            "image_id": selected_result['id']
        }

        # Calculate target output directory to check for existing data
        date_str_s1 = metadata_s1["date_time"]
        bbox_tuple = tuple(metadata_s1.get("bbox", []))
        area_id = abs(hash(bbox_tuple))
        target_output_dir = Path(project_root + "/data") / "S1" / date_str_s1 / str(area_id)

        expected_s1_files = ["VV.tif", "VH.tif"]
        if _check_existing_data(target_output_dir, expected_s1_files):
            print(f"Data for S1 at {date_str_s1} for area {area_id} already exists. Skipping download.")
            return {"status": "SKIPPED", "metadata": metadata_s1, "cost": 0.0}

        evalscript_s1_bands = '''
            //VERSION=3
            function setup() {
                return {
                    input: [{ bands: ["VV", "VH"] }],
                    output: [{id:"VV", bands:1, sampleType:SampleType.FLOAT32}, {id:"VH", bands:1, sampleType:SampleType.FLOAT32}]
                };
            }
            function evaluatePixel(sample) {
                return { VV:[sample.VV], VH:[sample.VH] };
            }
        '''

        date_str_s1 = metadata_s1["date_time"]
        s1_data_folder = Path(project_root + f"/data/S1/{date_str_s1}")
        s1_data_folder.mkdir(parents=True, exist_ok=True)

        request_s1_bands = SentinelHubRequest(
            data_folder=str(s1_data_folder),
            evalscript=evalscript_s1_bands,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL1_IW.define_from("s1grd", service_url=config.sh_base_url),
                    time_interval=(selected_result['properties']['datetime'], selected_result['properties']['datetime']),
                ),
            ],
            responses=[{"identifier": "VV", "format": {"type": "image/tiff"}}, {"identifier": "VH", "format": {"type": "image/tiff"}}],
            bbox=aoi_bbox,
            size=aoi_size,
            config=config,
        )
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with timer("Downloading S1 data"):
                    download_responses = request_s1_bands.get_data(save_data=True, decode_data=False)
                    cost_str = download_responses[0].headers['x-processingunits-spent']
                    cost_pu = float(cost_str)
                    print(f"Download successful. Copernicus PU cost: {cost_pu:.2f}")
                    return {"status": "SUCCESS", "metadata": metadata_s1, "cost": cost_pu}
            except Exception as e:
                print(f"Attempt {attempt + 1} failed with error: {e}")
                if attempt < max_retries - 1:
                    print("Waiting before retrying...")
                    time.sleep(5)
                else:
                    print("All download attempts failed for Sentinel-1.")
                    return {"status": "FAILED", "error_message": str(e)}
    except Exception as e:
        return {"status": "FAILED", "error_message": str(e)}



# Main execution block
if __name__ == "__main__":
    with open(project_root + "/config.json", "r") as f:
        config_run = json.load(f)
    print("Configuration loaded:", config_run)

    config = SHConfig()
    config.sh_client_id = config_run["client_ID"]
    config.sh_client_secret = config_run["client_secret"]
    config.sh_base_url = 'https://sh.dataspace.copernicus.eu'
    config.sh_token_url = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'

    # Define the API limit
    MAX_IMG_PIXELS = 2500

    total_pu_cost = 0
    bbox_list = config_run.get("bboxes", [])

    print(f"\nFound {len(bbox_list)} bounding box(es) to process.")

    for i, bbox_coords in enumerate(bbox_list):
        print(f"\n----- Starting on Bounding Box {i + 1} of {len(bbox_list)}: {bbox_coords} ----- ")
        
        # Convert raw coordinates to a BBox object
        aoi_bbox = BBox(bbox=bbox_coords, crs=CRS.WGS84)

        aoi_size = bbox_to_dimensions(aoi_bbox, resolution=config_run["resolution"])

        print(f"  Bounding Box (WGS84): {aoi_bbox.min_x:.4f}, {aoi_bbox.min_y:.4f}, {aoi_bbox.max_x:.4f}, {aoi_bbox.max_y:.4f}")
        print(f"  Resolution: {config_run["resolution"]}m")
        print(f"  Image Dimensions (pixels): {aoi_size[0]}x{aoi_size[1]}")
        
        # Safeguard check
        if aoi_size[0] == 0 or aoi_size[1] == 0:
             print(f"Warning: BBox {bbox_coords} resulted in 0 dimensions. Skipping this bounding box.")
             continue
        if aoi_size[0] > MAX_IMG_PIXELS or aoi_size[1] > MAX_IMG_PIXELS:
             print(f"Warning: BBox {bbox_coords} pixel dimensions {aoi_size} exceed limit. Skipping this bounding box.")
             continue

        '''s2_status = download_sentinel2(config, config_run, aoi_bbox, aoi_size, config_run["resolution"])
        if s2_status['status'] == 'SUCCESS':
            total_pu_cost += s2_status['cost']
            date_str_s2 = s2_status['metadata']["date_time"]
            organize_sentinel_downloads(project_root + f"/data/S2/{date_str_s2}", s2_status['metadata'])
        elif s2_status['status'] == 'SKIPPED':
            print(f"Skipping organization for S2 as data already exists.")
        else:
            print(f"Skipping organization for S2 due to download failure: {s2_status['error_message']}")'''

        s1_status = download_sentinel1(config, config_run, aoi_bbox, aoi_size, config_run["resolution"])
        if s1_status['status'] == 'SUCCESS':
            total_pu_cost += s1_status['cost']
            date_str_s1 = s1_status['metadata']["date_time"]
            organize_sentinel_downloads(project_root + f"/data/S1/{date_str_s1}", s1_status['metadata'])
        elif s1_status['status'] == 'SKIPPED':
            print(f"Skipping organization for S1 as data already exists.")
        else:
            print(f"Skipping organization for S1 due to download failure: {s1_status['error_message']}")

    # Run cleanup at the very end


    print(f"\nTotal Copernicus Processing Units spent for all bounding boxes: {total_pu_cost:.2f}")
    print("\nScript finished.")
