from flask import Flask, request, jsonify, render_template, make_response
import sqlite3
from cv_engine import analyze_medication
import csv
import io

app = Flask(__name__)
DB_FILE = "prescription.db"
MASTER_ADMIN_PASSWORD = "HACKATHON_ADMIN"

def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

# --- MEDICATION VERIFICATION ENDPOINT ---

@app.route('/api/verify', methods=['POST'])
def verify():
    data = request.get_json(force=True, silent=True) or {}
    rfid_uid = data.get('rfid_uid')
    measured_weight = float(data.get('weight_g', 0.0))
    shake_detected = bool(data.get('shake_detected', False)) # Triggered by hardware

    conn = get_db()
    cursor = conn.cursor()

    def log_and_return(status, cause):
        cursor.execute("INSERT INTO logs (rfid_uid, status, cause) VALUES (?, ?, ?)", 
                       (rfid_uid, status, cause))
        conn.commit()
        return jsonify({"status": status, "code": cause}), 200

    # 1. Hardware Tamper / Sensor Stability Gate
    if shake_detected:
        return log_and_return("REJECTED", "SENSOR_INSTABILITY_SHAKE_DETECTED")

    # 2. Patient Identity Gate
    patient = cursor.execute("SELECT * FROM patients WHERE rfid_uid = ?", (rfid_uid,)).fetchone()
    if not patient:
        return log_and_return("REJECTED", "PATIENT_NOT_FOUND")

    # 3. Vision Pipeline Execution
    vision = analyze_medication()
    if vision.get('status') != "SUCCESS":
        return log_and_return("REJECTED", vision.get('status', 'VISION_ERROR'))

    # 4. Partial Dosage / Half-Pill Check
    # If the weight is roughly half of the expected prescription, reject specifically
    half_weight = patient['expected_weight_g'] / 2.0
    if abs(measured_weight - half_weight) <= patient['tolerance_g']:
        return log_and_return("REJECTED", "PARTIAL_DOSAGE_HALF_PILL_DETECTED")

    # 5. Unregistered Color Check (e.g., Brown Tablet)
    detected_color = vision.get('color', '').lower()
    if detected_color not in ['white', 'red', 'blue', 'yellow']:
        return log_and_return("REJECTED", f"UNREGISTERED_COLOR_DETECTED_{detected_color.upper()}")

    # 6. Gravimetric (Weight) Check
    weight_delta = abs(measured_weight - patient['expected_weight_g'])
    if weight_delta > patient['tolerance_g']:
        return log_and_return("REJECTED", "DOSAGE_WEIGHT_MISMATCH")

    # 7. Visual Appearance Check (Color & Shape)
    if detected_color != patient['expected_color'].lower() or vision.get('shape', '').lower() != patient['expected_shape'].lower():
        return log_and_return("REJECTED", "VISUAL_APPEARANCE_MISMATCH")

    # 8. Physical Dimension Check (Size in mm)
    size_delta = abs(vision.get('size_mm', 0.0) - patient['expected_size_mm'])
    if size_delta > patient['size_tolerance_mm']:
        return log_and_return("REJECTED", "VISUAL_DIMENSION_MISMATCH")

    # ALL GATES PASSED
    return log_and_return("VERIFIED", "ALL_CHECKS_PASSED")

# --- UI & AUDIT ROUTES ---

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/logs', methods=['GET'])
def get_logs():
    conn = get_db()
    logs = conn.execute("SELECT * FROM logs ORDER BY timestamp DESC LIMIT 20").fetchall()
    return jsonify([dict(ix) for ix in logs])

@app.route('/api/logs/<int:log_id>', methods=['DELETE'])
def delete_log(log_id):
    conn = get_db()
    conn.execute("DELETE FROM logs WHERE id = ?", (log_id,))
    conn.commit()
    return jsonify({"status": "SUCCESS"}), 200

@app.route('/api/logs/clear', methods=['DELETE'])
def clear_all_logs():
    conn = get_db()
    conn.execute("DELETE FROM logs")
    conn.commit()
    return jsonify({"status": "SUCCESS"}), 200

# --- PATIENT PROFILES MANAGEMENT ---

@app.route('/patients')
def manage_patients():
    conn = get_db()
    patients = conn.execute("SELECT * FROM patients").fetchall()
    return render_template('patients.html', patients=patients)

@app.route('/api/patients/<rfid_uid>', methods=['DELETE'])
def delete_patient(rfid_uid):
    data = request.get_json(force=True, silent=True) or {}
    nurse_rfid = data.get('nurse_rfid')

    conn = get_db()
    nurse = conn.execute("SELECT * FROM nurses WHERE nurse_rfid = ?", (nurse_rfid,)).fetchone()
    if not nurse:
        return jsonify({"status": "REJECTED", "code": "UNAUTHORIZED_NURSE"}), 403

    conn.execute("DELETE FROM patients WHERE rfid_uid = ?", (rfid_uid,))
    conn.commit()
    return jsonify({"status": "SUCCESS"}), 200

@app.route('/register', methods=['GET', 'POST'])
def register_patient():
    if request.method == 'POST':
        conn = get_db()
        nurse_rfid = request.form.get('nurse_rfid')
        nurse = conn.execute("SELECT * FROM nurses WHERE nurse_rfid = ?", (nurse_rfid,)).fetchone()
        
        if not nurse:
            return render_template('register.html', error="UNAUTHORIZED: Invalid Nurse RFID Scan.")

        patient_id = request.form.get('patient_id')
        rfid_uid = request.form.get('rfid_uid')
        patient_name = request.form.get('patient_name')
        expected_weight_g = float(request.form.get('expected_weight_g', 0.0))
        tolerance_g = float(request.form.get('tolerance_g', 0.2))
        expected_color = request.form.get('expected_color', '').lower()
        expected_shape = request.form.get('expected_shape', '').lower()
        expected_size_mm = float(request.form.get('expected_size_mm', 0.0))
        size_tolerance_mm = float(request.form.get('size_tolerance_mm', 2.0))

        try:
            conn.execute('''
                INSERT INTO patients 
                (rfid_uid, patient_id, patient_name, expected_weight_g, tolerance_g, expected_color, expected_shape, expected_size_mm, size_tolerance_mm) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (rfid_uid, patient_id, patient_name, expected_weight_g, tolerance_g, expected_color, expected_shape, expected_size_mm, size_tolerance_mm))
            conn.commit()
            return render_template('register.html', success=f"Profile saved! Authorized by: {nurse['nurse_name']}")
        except sqlite3.IntegrityError:
            return render_template('register.html', error="Database Error: RFID Card or Patient ID is already registered.")
            
    return render_template('register.html')

# --- NURSE STAFF MANAGEMENT (ADMIN) ---

@app.route('/nurses', methods=['GET', 'POST'])
def manage_nurses():
    conn = get_db()
    if request.method == 'POST':
        master_pass = request.form.get('master_password')
        if master_pass != MASTER_ADMIN_PASSWORD:
            nurses = conn.execute("SELECT * FROM nurses").fetchall()
            return render_template('nurses.html', nurses=nurses, error="ACCESS DENIED: Invalid Master Password.")
        
        nurse_rfid = request.form.get('nurse_rfid')
        nurse_name = request.form.get('nurse_name')
        
        try:
            conn.execute("INSERT INTO nurses (nurse_rfid, nurse_name) VALUES (?, ?)", (nurse_rfid, nurse_name))
            conn.commit()
            nurses = conn.execute("SELECT * FROM nurses").fetchall()
            return render_template('nurses.html', nurses=nurses, success=f"Admin Success: {nurse_name} registered.")
        except sqlite3.IntegrityError:
            nurses = conn.execute("SELECT * FROM nurses").fetchall()
            return render_template('nurses.html', nurses=nurses, error="Database Error: Nurse RFID is already registered.")
            
    nurses = conn.execute("SELECT * FROM nurses").fetchall()
    return render_template('nurses.html', nurses=nurses)

@app.route('/api/nurses/<nurse_rfid>', methods=['DELETE'])
def delete_nurse(nurse_rfid):
    data = request.get_json(force=True, silent=True) or {}
    master_pass = data.get('master_password')

    if master_pass != MASTER_ADMIN_PASSWORD:
        return jsonify({"status": "REJECTED", "code": "UNAUTHORIZED_ADMIN"}), 403

    conn = get_db()
    conn.execute("DELETE FROM nurses WHERE nurse_rfid = ?", (nurse_rfid,))
    conn.commit()
    return jsonify({"status": "SUCCESS"}), 200




@app.route('/api/logs/download', methods=['GET'])
def download_logs():
    conn = get_db()
    logs = conn.execute("SELECT timestamp, rfid_uid, status, cause FROM logs ORDER BY timestamp DESC").fetchall()

    # Create CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)
    
    # CSV Header
    writer.writerow(['Timestamp', 'Patient RFID', 'Status', 'Diagnostic Cause'])
    
    # CSV Rows
    for log in logs:
        writer.writerow([log['timestamp'], log['rfid_uid'], log['status'], log['cause']])
    
    output.seek(0)
    
    # Prepare response as a downloadable CSV attachment
    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=clinical_guard_audit_logs.csv"
    response.headers["Content-type"] = "text/csv"
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)