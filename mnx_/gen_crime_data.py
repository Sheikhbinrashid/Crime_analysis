import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ── Configuration ─────────────────────────────────────────────────────────────
NUM_RECORDS = 2000
START_DATE  = datetime(2025, 1, 1)
END_DATE    = datetime(2025, 5, 20)

CRIME_TYPES = [
    "Theft", "Assault", "Burglary", "Robbery",
    "Vandalism", "Drug Offense", "Fraud"
]

DISTRICTS = [
    "Harare", "Bulawayo", "Mutare", "Gweru",
    "Chitungwiza", "Kadoma", "Marondera", "Masvingo"
]

# Approximate bounding box for Zimbabwe
LAT_MIN, LAT_MAX = -22.0, -15.0
LON_MIN, LON_MAX =  25.0,  34.0

# ── Generate Data ─────────────────────────────────────────────────────────────
def random_date(start: datetime, end: datetime) -> datetime:
    """Return a random datetime between start and end."""
    delta = end - start
    random_days = np.random.randint(0, delta.days + 1)
    return start + timedelta(days=random_days)

records = []
for _ in range(NUM_RECORDS):
    date     = random_date(START_DATE, END_DATE).strftime("%Y-%m-%d")
    crime    = np.random.choice(CRIME_TYPES)
    district = np.random.choice(DISTRICTS)
    lat      = np.round(np.random.uniform(LAT_MIN, LAT_MAX), 6)
    lon      = np.round(np.random.uniform(LON_MIN, LON_MAX), 6)
    records.append((date, crime, lat, lon, district))

df = pd.DataFrame(
    records,
    columns=["Date", "Crime_Type", "Latitude", "Longitude", "District"]
)

# ── Save to CSV ────────────────────────────────────────────────────────────────
df.to_csv("crime_data.csv", index=False)
print(f"Generated {NUM_RECORDS} records → crime_data.csv")
