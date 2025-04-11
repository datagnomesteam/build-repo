import json
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime
import os
from db_info import get_db_config, get_db_connection, get_db_cursor

def create_database():
    """Create the database and tables if they don't exist"""
    # connect to PostgreSQL server (to create the database if needed)
    conn = get_db_connection()
    conn.autocommit = True
    cursor = conn.cursor()

    # sanity check - drop database
    cursor.execute(f"DROP DATABASE {get_db_config()['database']} WITH (FORCE)") # clear database if exists
    
    # create database if it doesn't exist
    cursor.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (get_db_config()["database"],))
    if not cursor.fetchone():
        print(f"Creating database {get_db_config()['database']}...")
        cursor.execute(f"CREATE DATABASE {get_db_config()['database']}")
    cursor.close()
    conn.close()
    
    # connect to the database and create tables from schema
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # # get list of all files in directory and execute sql where applicable
    # files = os.listdir('../db')
    # for file in files:
    #     if '.sql' not in file:
    #         continue
    #     # read and execute sql
    #     schema_sql = open(f'../db/{file}', 'r').read()
    #     cursor.execute(schema_sql)
    #     conn.commit()

    #     print(f'{file} loaded successfully.')

    with open("../db/device_event.sql", "r") as f:
        sql = f.read()
    cursor.execute(sql)
    conn.commit()
    print(f'{f} loaded successfully')

    with open("../db/device.sql", "r") as f:
        sql = f.read()
    cursor.execute(sql)
    conn.commit()
    print(f'{f} loaded successfully')

    with open("../db/recallenforcement.sql", "r") as f:
        sql = f.read()
    cursor.execute(sql)
    conn.commit()
    print(f'{f} loaded successfully')

    with open("../db/classification.sql", "r") as f:
        sql = f.read()
    cursor.execute(sql)
    conn.commit()
    print(f'{f} loaded successfully')

    with open("../db/patient.sql", "r") as f:
        sql = f.read()
    cursor.execute(sql)
    conn.commit()
    print(f'{f} loaded successfully')

    with open("../db/mdr_text.sql", "r") as f:
        sql = f.read()
    cursor.execute(sql)
    conn.commit()
    print(f'{f} loaded successfully')

    with open("../db/510k.sql", "r") as f:
        sql = f.read()
    cursor.execute(sql)
    conn.commit()
    print(f'{f} loaded successfully')

    with open("../db/premarketapproval.sql", "r") as f:
        sql = f.read()
    cursor.execute(sql)
    conn.commit()
    print(f'{f} loaded successfully')

    with open("../db/recalls.sql", "r") as f:
        sql = f.read()
    cursor.execute(sql)
    conn.commit()
    print(f'{f} loaded successfully')

    cursor.close()
    conn.close()

def parse_date(date_str):
    """Parse date string to PostgreSQL compatible date or None"""
    if not date_str:
        return None
    return date_str

def import_recall_data(json_file_path):
    """Import recall data from JSON file into PostgreSQL database"""
    # load json data
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # connect to database
    conn = get_db_connection()
    cursor = get_db_cursor(conn)
    
    # metadata
    meta = data.get('meta', {})
    meta_data = {
        'disclaimer': meta.get('disclaimer'),
        'license': meta.get('license'),
        'last_updated': parse_date(meta.get('last_updated'))
    }
    
    # insert recall data
    recall_results = data.get('results', [])
    total_recalls = len(recall_results)
    
    print(f"Processing {total_recalls} recalls...")
    
    for i, recall in enumerate(recall_results):
        # main recall data
        insert_recall_sql = """
        INSERT INTO recall (
            cfres_id, event_date_initiated, event_date_posted, recall_status, res_event_number,
            product_code, product_description, code_info, product_res_number, recalling_firm,
            address_1, city, state, postal_code, additional_info_contact, reason_for_recall,
            root_cause_description, action, product_quantity, distribution_pattern
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """
        
        cursor.execute(insert_recall_sql, (
            recall.get('cfres_id'),
            parse_date(recall.get('event_date_initiated')),
            parse_date(recall.get('event_date_posted')),
            recall.get('recall_status'),
            recall.get('res_event_number'),
            recall.get('product_code'),
            recall.get('product_description'),
            recall.get('code_info'),
            recall.get('product_res_number'),
            recall.get('recalling_firm'),
            recall.get('address_1'),
            recall.get('city'),
            recall.get('state'),
            recall.get('postal_code'),
            recall.get('additional_info_contact'),
            recall.get('reason_for_recall'),
            recall.get('root_cause_description'),
            recall.get('action'),
            recall.get('product_quantity'),
            recall.get('distribution_pattern')
        ))
        
        recall_id = cursor.fetchone()[0]
        
        # k_numbers
        if 'k_numbers' in recall and recall['k_numbers']:
            k_numbers_data = [(recall_id, k_number) for k_number in recall['k_numbers']]
            execute_values(
                cursor,
                "INSERT INTO recall_k_numbers (recall_id, k_number) VALUES %s",
                k_numbers_data
            )
        
        # OpenFDA data if available
        if 'openfda' in recall and recall['openfda']:
            openfda = recall['openfda']
            
            # Insert base OpenFDA data
            insert_openfda_sql = """
            INSERT INTO recall_openfda (
                recall_id, device_class, device_name, medical_specialty_description
            ) VALUES (%s, upper(%s), %s, %s)
            RETURNING id
            """
            
            cursor.execute(insert_openfda_sql, (
                recall_id,
                openfda.get('device_class'),
                openfda.get('device_name'),
                openfda.get('medical_specialty_description')
            ))
            
            openfda_id = cursor.fetchone()[0]
            
            # Insert OpenFDA k_numbers
            if 'k_number' in openfda and openfda['k_number']:
                openfda_k_numbers_data = [(openfda_id, k_number) for k_number in openfda['k_number']]
                execute_values(
                    cursor,
                    "INSERT INTO recall_openfda_k_numbers (recall_openfda_id, k_number) VALUES %s",
                    openfda_k_numbers_data
                )
            
            # Insert OpenFDA registration_numbers
            if 'registration_number' in openfda and openfda['registration_number']:
                reg_numbers_data = [(openfda_id, reg_number) for reg_number in openfda['registration_number']]
                execute_values(
                    cursor,
                    "INSERT INTO recall_openfda_registration_numbers (recall_openfda_id, registration_number) VALUES %s",
                    reg_numbers_data
                )
            
            # Insert OpenFDA fei_numbers
            if 'fei_number' in openfda and openfda['fei_number']:
                fei_numbers_data = [(openfda_id, fei_number) for fei_number in openfda['fei_number']]
                execute_values(
                    cursor,
                    "INSERT INTO recall_openfda_fei_numbers (recall_openfda_id, fei_number) VALUES %s",
                    fei_numbers_data
                )
            
            # Insert regulation_number if available
            if 'regulation_number' in openfda:
                regulation_numbers_data = [(openfda_id, openfda['regulation_number'])]
                execute_values(
                    cursor,
                    "INSERT INTO recall_openfda_regulation_numbers (recall_openfda_id, regulation_number) VALUES %s",
                    regulation_numbers_data
                )
        
        # meta data for this recall
        cursor.execute(
            "INSERT INTO recall_meta (recall_id, disclaimer, license, last_updated) VALUES (%s, %s, %s, %s) RETURNING id",
            (recall_id, meta_data['disclaimer'], meta_data['license'], meta_data['last_updated'])
        )
        
        meta_id = cursor.fetchone()[0]
        
        # meta results
        meta_results = meta.get('results', {})
        cursor.execute(
            "INSERT INTO recall_meta_results (recall_meta_id, skip, limits, total) VALUES (%s, %s, %s, %s)",
            (meta_id, meta_results.get('skip'), meta_results.get('limit'), meta_results.get('total'))
        )
        
        # print progress
        if (i + 1) % 10 == 0 or i == total_recalls - 1:
            print(f"Processed {i + 1}/{total_recalls} recalls...")
    
    conn.commit()
    cursor.close()
    conn.close()
    
    print(f"successfully imported {total_recalls} recalls.")

def main():
    json_file_path = "recalls-shortened.json"
    
    try:
        print("setting up database...")
        create_database()
        
        # print("importing recall data...")
        # import_recall_data(json_file_path)
        
        # print("data import completed successfully!")

    except Exception as e:
        print(f"Error (main): {e}")

if __name__ == "__main__":
    main()