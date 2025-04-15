
import psycopg2
import dotenv
import os
dotenv.load_dotenv()

def get_db_config():
    return {
        "host": os.getenv("DB_HOST") or "localhost",
        "database": os.getenv("DB_NAME") or "cse6242",
        "user": os.getenv("DB_USER") or "willbaldwin",
        "password": os.getenv("DB_PASSWORD") or "password123",
        "port": os.getenv("DB_PORT") or 5432
    }

def get_db_connection():
    return psycopg2.connect(**get_db_config())

def get_db_cursor(conn):
    return conn.cursor()