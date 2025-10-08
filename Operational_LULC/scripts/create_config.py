import json
import os
import math

project_root = '/home/envitwin/Desktop/venvs/EnviTwin/Operational_LULC'

config = {
    "client_ID": "sh-0f947091-c7b3-43c5-8c1f-07c03da3bae7",
    "client_secret": "oISz9fmQJ1qleO7u6CsgHcTUZZmZbflD",
    "mosaickingOrderS2": "leastCC", # "mostRecent", "leastCC", "leastRecent"
    "mosaickingOrderS1": "mostRecent",
    "resolution": 10,
    "S2_time_interval": ("2025-09-01", "2025-10-08"),
    "S1_time_interval": ("2025-09-18", "2025-09-20"),
}

config["bboxes"] = [
    [22.815966388989274, 40.5461501866596, 23.01825285288775, 40.707953967477884]
]

with open(project_root + "/config.json", "w") as f:
    json.dump(config, f, indent=4)

print(f"Successfully created config.json with {len(config["bboxes"])} bounding box(es).")
