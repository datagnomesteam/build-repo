import psycopg2
from configs import db_credentials

try:
    # connection parameters
    conn_params = {
        "dbname": db_credentials['database'],
        "user": db_credentials['username'],
        "password": db_credentials['password'],
        "host": db_credentials['host'],
        "port": db_credentials['port']
    }

    # establish the connection
    conn = psycopg2.connect(**conn_params)
    cur = conn.cursor()

    # list of SQL statements to create views
    view_statements = [
        """
        DROP VIEW integrated_device_view;
        """,
        """
        DROP VIEW integrated_manufacturer_view;
        """,
        """
        DROP VIEW integrated_manufacturer_address_view;
        """,
        """
        CREATE OR REPLACE VIEW integrated_device_view AS
        SELECT device_name, manufacturer_name, device_class, regulation_number, medical_specialty_description, SUM(death) as deaths, SUM(injury) as injuries, SUM(malfunction) as malfunctions, SUM(other) as other, SUM(recall) as recalls, SUM(DISTINCT("1")) as "class_1", SUM(DISTINCT("2")) as "class_2", SUM(DISTINCT("3")) as "class_3", COUNT(device_name) as records, COUNT(DISTINCT(pma_number)) as pma_submissions, COUNT(DISTINCT(k_number)) as k_submissions 
        FROM integrated
        GROUP BY device_name, manufacturer_name, device_class, regulation_number, medical_specialty_description
        ORDER BY manufacturer_name, device_class, device_name;
        """,
        """
        CREATE OR REPLACE VIEW integrated_manufacturer_view AS
        SELECT manufacturer_name, SUM(death) as deaths, SUM(injury) as injuries, SUM(malfunction) as malfunctions, SUM(other) as other, SUM(recall) as recalls, SUM(DISTINCT("1")) as "class_1", SUM(DISTINCT("2")) as "class_2", SUM(DISTINCT("3")) as "class_3", COUNT(DISTINCT(device_name)) as unique_devices, COUNT(DISTINCT(pma_number)) as pma_submissions, COUNT(DISTINCT(k_number)) as k_submissions
        FROM integrated
        GROUP BY manufacturer_name
        ORDER BY manufacturer_name;
        """,
        """
        CREATE OR REPLACE VIEW integrated_manufacturer_address_view AS
        SELECT DISTINCT manufacturer_name, manufacturer_street, manufacturer_city, manufacturer_state, manufacturer_country, manufacturer_postal_code
        FROM integrated
        ORDER BY manufacturer_name;
        """
    ]

    for view_sql in view_statements:
        try:
            cur.execute(view_sql)
            conn.commit()
            print("View created successfully.")
        except psycopg2.DatabaseError as db_err:
            print(f"Error creating view: {db_err}")
            conn.rollback()
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            conn.rollback()

except psycopg2.DatabaseError as db_err:
    print(f"Database connection error: {db_err}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
finally:
    if cur:
        cur.close()
    if conn:
        conn.close()

