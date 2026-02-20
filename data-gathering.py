import csv
import os
from datacanvas import DataCanvas, SortOrder
from dotenv import load_dotenv

load_dotenv()

# Initialise SDK
client = DataCanvas(
    access_key_client=os.environ["DATACANVAS_ACCESS_KEY_ID"],
    access_key_secret=os.environ["DATACANVAS_SECRET_KEY"],
    project_id=int(os.environ["DATACANVAS_PROJECT_ID"]),
    base_url=os.environ["DATACANVAS_BASE_URL"],
)

# List all devices
devices = client.devices.list()
print(f"Found {len(devices.devices)} devices")

PAGE_SIZE = 1000
page = 0
all_rows = []
fieldnames = None

while True:
    data = client.data.list(
        table_name="weatherdatanew",
        devices=[12],
        page=page,
        limit=PAGE_SIZE,
        order=SortOrder.DESC,
    )

    if page == 0:
        print(f"Total data points: {data.count}")

    # Collect rows from this page
    page_rows = []
    for device_id, points in data.data.items():
        for point in points:
            row = {"id": point.id, "device": point.device, **point.extra}
            page_rows.append(row)

    if not page_rows:
        break

    if fieldnames is None:
        fieldnames = list(page_rows[0].keys())

    all_rows.extend(page_rows)
    print(f"  Fetched page {page} — {len(all_rows)}/{data.count} rows so far")

    if len(all_rows) >= data.count:
        break

    page += 1

# Save data to CSV
# Output path matches pipeline.py RAW_DATA_PATH = "raw_data.csv"
OUT_PATH = os.environ.get("RAW_DATA_PATH", "raw_data.csv")
if all_rows:
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Saved {len(all_rows)} rows to {OUT_PATH}")
else:
    print("No data to save")