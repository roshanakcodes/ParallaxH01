import sqlite3

conn = sqlite3.connect("prescription.db")
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS patients (
    rfid_uid TEXT PRIMARY KEY,
    patient_id TEXT,
    patient_name TEXT,
    expected_weight_g REAL,
    tolerance_g REAL,
    expected_color TEXT,
    expected_shape TEXT,
    expected_size_mm REAL,
    size_tolerance_mm REAL
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS nurses (
    nurse_rfid TEXT PRIMARY KEY,
    nurse_name TEXT
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rfid_uid TEXT,
    status TEXT,
    cause TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')

cursor.execute("INSERT OR IGNORE INTO nurses (nurse_rfid, nurse_name) VALUES (?, ?)", ('NURSE_123', 'Default Admin Nurse'))

conn.commit()
conn.close()
print("Database schema initialized successfully!")