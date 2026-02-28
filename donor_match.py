import sqlite3
import pandas as pd
from math import radians, sin, cos, asin, sqrt

DB_PATH = "donors.db"

COMPATIBILITY = {
    "A+": ["A+","A-","O+","O-"],
    "A-": ["A-","O-"],
    "B+": ["B+","B-","O+","O-"],
    "B-": ["B-","O-"],
    "AB+": ["A+","A-","B+","B-","AB+","AB-","O+","O-"],
    "AB-": ["AB-","A-","B-","O-"],
    "O+": ["O+","O-"],
    "O-": ["O-"]
}

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2-lat1)
    dlon = radians(lon2-lon1)

    a = sin(dlat/2)**2 + cos(radians(lat1))* \
        cos(radians(lat2))*sin(dlon/2)**2

    return 2 * R * asin(sqrt(a))


def match_donors(blood, lat, lon, top_k=5):

    conn = sqlite3.connect(DB_PATH)

    compatible = tuple(COMPATIBILITY[blood])

    query = f"""
    SELECT d.donor_id, d.name, d.blood_group,
           d.location, d.latitude, d.longitude,
           m.hemoglobin, m.chronic_conditions
    FROM donors d
    JOIN medical_logs m
    ON d.donor_id = m.donor_id
    WHERE d.available = 1
    AND m.hemoglobin > 12.5
    AND d.blood_group IN {compatible}
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        return df

    df["distance_km"] = df.apply(
        lambda x: haversine(lat, lon,
                            x["latitude"], x["longitude"]),
        axis=1
    )

    df = df.sort_values("distance_km").head(top_k)

    # Privacy masking
    df["name"] = "Hidden (Admin Access Required)"
    df["chronic_conditions"] = "Protected Medical Record"

    return df.reset_index(drop=True)