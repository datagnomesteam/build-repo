import json
import psycopg2
from psycopg2.extras import execute_values

# Database connection parameters
DB_PARAMS = {
    "dbname": "your_database",
    "user": "your_user",
    "password": "your_password",
    "host": "your_host",
    "port": "your_port"
}

def insert_device_event(cursor, data):
    query = """
        INSERT INTO device_event (
            adverse_event_flag, event_type, report_number, date_received, date_of_event, report_source_code, 
            product_problem_flag, reporter_occupation_code, manufacturer_link_flag, health_professional
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
    """
    values = (
        data.get("adverse_event_flag"),
        data.get("event_type"),
        data.get("report_number"),
        data.get("date_received"),
        data.get("date_of_event"),
        data.get("report_source_code"),
        data.get("product_problem_flag"),
        data.get("reporter_occupation_code"),
        data.get("manufacturer_link_flag"),
        data.get("health_professional"),
    )
    cursor.execute(query, values)
    return cursor.fetchone()[0]  # Get the inserted ID

def insert_patient(cursor, event_id, patient_data):
    query = """
        INSERT INTO patient (event_number, date_received, patient_sequence_number, patient_age, patient_sex, patient_weight, patient_ethnicity, patient_race)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
    """
    values = (
        event_id,
        patient_data.get("date_received"),
        patient_data.get("patient_sequence_number"),
        patient_data.get("patient_age"),
        patient_data.get("patient_sex"),
        patient_data.get("patient_weight"),
        patient_data.get("patient_ethnicity"),
        patient_data.get("patient_race"),
    )
    cursor.execute(query, values)
    return cursor.fetchone()[0]

def insert_patient_problems(cursor, patient_id, problems):
    query = "INSERT INTO patient_problem (patient_id, problem_description) VALUES %s"
    values = [(patient_id, problem) for problem in problems]
    execute_values(cursor, query, values)

def insert_mdr_text(cursor, text_data):
    query = """
        INSERT INTO mdr_text (date_report, mdr_text_key, patient_sequence_number, text, text_type_code)
        VALUES (%s, %s, %s, %s, %s);
    """
    values = (
        text_data.get("date_report"),
        text_data.get("mdr_text_key"),
        text_data.get("patient_sequence_number"),
        text_data.get("text"),
        text_data.get("text_type_code"),
    )
    cursor.execute(query, values)

def main():
    with open("data.json") as f:
        data = json.load(f)
    
    conn = psycopg2.connect(**DB_PARAMS)
    cursor = conn.cursor()
    
    try:
        # Insert device event
        event_id = insert_device_event(cursor, data)
        
        # Insert patients
        for patient in data.get("patient", []):
            patient_id = insert_patient(cursor, event_id, patient)
            
            # Insert patient problems
            if "patient_problems" in patient:
                insert_patient_problems(cursor, patient_id, patient["patient_problems"])
        
        # Insert MDR text
        for mdr in data.get("mdr_text", []):
            insert_mdr_text(cursor, mdr)
        
        conn.commit()
    except Exception as e:
        conn.rollback()
        print("Error:", e)
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    main()


import json
import psycopg2
from datetime import datetime

# Database connection parameters
DB_PARAMS = {
    "dbname": "your_database",
    "user": "your_user",
    "password": "your_password",
    "host": "localhost",
    "port": "5432"
}

# Load JSON data
with open("data.json", "r") as f:
    data = json.load(f)

# Extract main device event data
device_event_values = [
    (
        data.get("adverse_event_flag"),
        data.get("date_of_event"),
        data.get("date_received"),
        data.get("date_report"),
        data.get("event_location"),
        data.get("event_type"),
        data.get("health_professional"),
        data.get("initial_report_to_fda"),
        data.get("manufacturer_name"),
        data.get("manufacturer_city"),
        data.get("manufacturer_state"),
        data.get("manufacturer_zip_code"),
        data.get("mdr_report_key"),
        data.get("number_devices_in_event"),
        data.get("product_problem_flag"),
        data.get("report_number"),
        data.get("report_source_code"),
        data.get("reporter_occupation_code"),
    )
    for result in data['results']
]

# Define SQL query with placeholders
insert_query = """
INSERT INTO device_event (
    adverse_event_flag, date_of_event, date_received, date_report, 
    event_location, event_type, health_professional, initial_report_to_fda, 
    manufacturer_name, manufacturer_city, manufacturer_state, manufacturer_zip_code, 
    mdr_report_key, number_devices_in_event, product_problem_flag, report_number, 
    report_source_code, reporter_occupation_code
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

# Connect to the database and execute batch insert
try:
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    
    cur.executemany(insert_query, device_event_values)
    
    conn.commit()  # Save changes
    print("Data inserted successfully!")
    
    cur.close()
    conn.close()
except Exception as e:
    print("Error inserting data:", e)
