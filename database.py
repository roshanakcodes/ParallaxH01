import sqlite3

def init_db():
    conn = sqlite3.connect("prescription.db")
    cursor = conn.cursor()

    # 1. Patient Profiles (Includes new size metrics)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            rfid_uid TEXT PRIMARY KEY,
            patient_id TEXT UNIQUE,
            patient_name TEXT,
            expected_weight_g REAL,
            tolerance_g REAL,
            expected_color TEXT,
            expected_shape TEXT,
            expected_size_mm REAL,
            size_tolerance_mm REAL
        )
    ''')

    # 2. Nurse Authorization Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS nurses (
            nurse_rfid TEXT PRIMARY KEY,
            nurse_name TEXT
        )
    ''')

    # 3. Nurse Dashboard Logs
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            rfid_uid TEXT,
            status TEXT,
            cause TEXT
        )
    ''')

    # Insert a dummy Nurse for your demo
    cursor.execute("INSERT OR IGNORE INTO nurses (nurse_rfid, nurse_name) VALUES ('NURSE_123', 'Head Nurse Sarah')")

    conn.commit()
    conn.close()
    print("Database re-initialized with Dimensions and Nurse Auth successfully.")

if __name__ == "__main__":
    init_db()