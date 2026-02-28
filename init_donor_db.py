# =====================================================
# Life-Link Donor Database Initialization
# =====================================================

import sqlite3
from datetime import datetime
import random
from cryptography.fernet import Fernet

# -----------------------------------------------------
# ENCRYPTION SETUP (Data Anonymization)
# -----------------------------------------------------
KEY = b'c29tZV9zZWN1cmVfa2V5X2Zvcl9kZW1vX3Byb2plY3Q='
cipher = Fernet(KEY)


def encrypt_data(text):
    return cipher.encrypt(text.encode()).decode()


# -----------------------------------------------------
# DATABASE CONNECTION
# -----------------------------------------------------
conn = sqlite3.connect("donors.db")
cursor = conn.cursor()

# -----------------------------------------------------
# USER PROFILES TABLE
# -----------------------------------------------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS donors (
    donor_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    blood_group TEXT,
    age INTEGER,
    location TEXT,
    latitude REAL,
    longitude REAL,
    available INTEGER
)
""")

# -----------------------------------------------------
# MEDICAL LOGS TABLE
# -----------------------------------------------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS medical_logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    donor_id INTEGER,
    hemoglobin REAL,
    chronic_conditions TEXT,
    timestamp TEXT,
    FOREIGN KEY(donor_id) REFERENCES donors(donor_id)
)
""")

# -----------------------------------------------------
# ANDHRA PRADESH LOCATIONS
# -----------------------------------------------------
cities = [
    ("Vijayawada",16.5062,80.6480),
    ("Guntur",16.3067,80.4365),
    ("Tenali",16.2430,80.6400),
    ("Eluru",16.7107,81.0952),
    ("Rajahmundry",17.0005,81.8040),
    ("Kakinada",16.9891,82.2475),
    ("Nellore",14.4426,79.9865),
    ("Tirupati",13.6288,79.4192),
    ("Ongole",15.5057,80.0499),
    ("Visakhapatnam",17.6868,83.2185),
    ("Anantapur",14.6819,77.6006),
    ("Kadapa",14.4673,78.8242),
    ("Kurnool",15.8281,78.0373),
    ("Chittoor",13.2172,79.1003),
]

blood_groups = ["O+","O-","A+","A-","B+","B-","AB+","AB-"]

names = [
    "Ravi","Suresh","Kiran","Anitha","Lakshmi","Prasad",
    "Divya","Venkatesh","Meena","Harsha","Ramesh",
    "Teja","Naveen","Deepika","Sowmya","Karthik",
    "Bhavani","Arjun","Keerthi","Mahesh","Pavan"
]

# -----------------------------------------------------
# INSERT DONORS
# -----------------------------------------------------
donor_records = []

for i in range(60):

    name = random.choice(names) + f"_{i}"
    blood = random.choice(blood_groups)
    age = random.randint(18,45)
    city = random.choice(cities)

    donor_records.append(
        (name, blood, age, city[0], city[1], city[2], 1)
    )

cursor.executemany("""
INSERT INTO donors
(name,blood_group,age,location,latitude,longitude,available)
VALUES (?,?,?,?,?,?,?)
""", donor_records)

# -----------------------------------------------------
# INSERT ENCRYPTED MEDICAL LOGS
# -----------------------------------------------------
cursor.execute("SELECT donor_id FROM donors")
ids = cursor.fetchall()

logs = []

for (donor_id,) in ids:

    hb = round(random.uniform(12.6,15.5),1)

    condition = encrypt_data(
        random.choice(["None","Asthma","Diabetes"])
    )

    logs.append(
        (donor_id, hb, condition, datetime.now().isoformat())
    )

cursor.executemany("""
INSERT INTO medical_logs
(donor_id,hemoglobin,chronic_conditions,timestamp)
VALUES (?,?,?,?)
""", logs)

conn.commit()
conn.close()

print("✅ Secure Life-Link donor database created successfully.")