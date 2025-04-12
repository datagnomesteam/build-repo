import psycopg2
from configs import db_credentials

def try_query(query):
    try:
        cur.execute(query)
        conn.commit()
        print(f"{query} executed successfully.")
    except psycopg2.DatabaseError as db_err:
        print(f"Error creating table: {db_err}")
        conn.rollback()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        conn.rollback()

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

    # list of source to sample
    sample_sources = ["device", "device_classification", "device_enforcements", "device_event", "mdr_text", "medical_specialty", "patient", "patient_problem", "pma_submission", "recall", "submission"]

    for source in sample_sources:
        sample_query = f"CREATE TABLE {source}_sample AS SELECT * FROM {source} TABLESAMPLE SYSTEM (10) REPEATABLE (12345);"
        drop_query = f"DROP TABLE {source} CASCADE;"
        alter_query = f"ALTER TABLE {source}_sample RENAME TO {source}"
        try_query(sample_query)
        try_query(drop_query)
        try_query(alter_query)

except psycopg2.DatabaseError as db_err:
    print(f"Database connection error: {db_err}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
finally:
    if cur:
        cur.close()
    if conn:
        conn.close()

