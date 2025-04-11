#this file loads all the data into the database
import os
import orjson
import json
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
import uuid
import init_db
import io
import threading
import psutil
import atexit
import time
from db_info import get_db_config, get_db_connection, get_db_cursor


"""
NOTE(s) (Priya)
Make sure to pip install orjson

For this sample I am using postgres 17 locally
-- add postgres lib and bin to PATH
-- to query db in cmd: 
        $ psql -U postgres
        $ \c openfda_recalls
        $ \dt  --> shows tables
        $ \q --> quits db

This script will delete existing db and reinitilize. It will load data from 1 device-event & recall json file.
-- select count(*) from device_event; -- expected 100000
-- select count(*) from recall; -- expected 54520

estimated runtime: ~1.5-2 mins
"""

datapath = "sample-data/"

connection_pool = ThreadedConnectionPool(1, 50, **get_db_config())

def list_files_recursively(directory):
    fileList = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if("zip" in file):
                continue
            if("pdf" in file):
                continue
            fileList.append(root+"/"+file)
    return fileList

def device_classification_import(file):
    """
    relevant files:
    classification.sql
    """
    with open(file, 'rb') as f:
        data = orjson.loads(f.read())

    conn = connection_pool.getconn()
    cur = conn.cursor()

    device_classification_values = []
    openfda_device_classification_values = []
    for result in data['results']:
        id = str(uuid.uuid4())
        device_classification_values.append((
            id,
            result.get("definition"),
            result.get("device_name"),
            result.get("device_class"),
            result.get("gmp_exempt_flag"),
            result.get("implant_flag"),
            result.get("life_sustain_support_flag"),
            result.get("medical_specialty_code"),
            result.get("product_code"),
            result.get("regulation_number"),
            result.get("review_code"),
            result.get("review_panel"),
            result.get("submission_type_id"),
            result.get("summary_malfunction_reporting"),
            result.get("third_party_flag"),
            result.get("unclassified_reason")
        ))

        openfda = result["openfda"]
        openfda_device_classification_values.append((
            id,
            openfda.get("pma_number"),
            openfda.get("fei_number"),
            openfda.get("k_number"),
            openfda.get("registration_number"),
        ))

    # insert device classification values
    #query = """
    #INSERT INTO device_classification (
    #    id, definition, device_class, device_name, gmp_exempt_flag, implant_flag, life_sustain_support_flag,
    #    medical_specialty_code, product_code, regulation_number, review_code, review_panel, submission_type_id,
    #    summary_malfunction_reporting, third_party_flag, unclassified_reason
    #) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    #"""
    #cur.executemany(query, device_classification_values)
    #conn.commit()

    data_stream = io.StringIO()
    for row in device_classification_values:
        formatted_row = []
        for value in row:
            if value is None:
                formatted_row.append('\\N')  # PostgreSQL NULL
            elif value in ["", "\\"]:
                formatted_row.append("\\N")
            elif isinstance(value, list):
                formatted_row.append("{" + ",".join(f'"{v}"' for v in value) + "}")  # Convert lists to comma-separated values
            elif isinstance(value, str):
                formatted_row.append(value.replace("\\", "\\\\").replace("\t", " ").replace("\n", " "))
            else:
                formatted_row.append(str(value).replace("\t", " ").replace("\n", " "))  # Convert everything else to string
        data_stream.write("\t".join(formatted_row) + "\n") 
    data_stream.seek(0)

    cur.copy_from(data_stream, 'device_classification', sep="\t", 
              columns=[
                  "id", "definition", "device_class", "device_name", "gmp_exempt_flag", "implant_flag", 
                  "life_sustain_support_flag", "medical_specialty_code", "product_code", "regulation_number", 
                  "review_code", "review_panel", "submission_type_id", "summary_malfunction_reporting", 
                  "third_party_flag", "unclassified_reason"
              ])
    conn.commit()

    # insert openfda device classification values
    query = """
    INSERT INTO openfda_device_classification (
        device_classification_id, pma_number, fei_number, k_number, registration_number
    ) VALUES (%s, %s, %s, %s, %s)
    """
    cur.executemany(query, openfda_device_classification_values)
    conn.commit()

    cur.close()
    connection_pool.putconn(conn)

def device_enforcement_import(file):
    """
    relevant files:
    recallenforcement.sql
    """
    with open(file, 'rb') as f:
        data = orjson.loads(f.read())

    conn = connection_pool.getconn()
    cur = conn.cursor()

    # Extract recall enforcement data
    enforcement_values = []
    for result in data['results']:
        id = str(uuid.uuid4())
        enforcement_values.append((
            id,
            result.get("address_1"),
            result.get("address_2"),
            result.get("center_classification_date"),
            result.get("city"),
            result.get("classification"),
            result.get("code_info"),
            result.get("country"),
            result.get("distribution_pattern"),
            result.get("event_id"),            
            result.get("initial_firm_notification"),
            result.get("more_code_info"),
            result.get("product_code"),
            result.get("product_description"),
            result.get("product_quantity"),
            result.get("product_type"),
            result.get("reason_for_recall"),
            result.get("recall_initiation_date"),
            result.get("recall_number"),
            result.get("recalling_firm"),
            result.get("report_date"),
            result.get("state"),
            result.get("status"),
            result.get("termination_date"),
            result.get("voluntary_mandated")
        ))

    # insert values
    query = """
    INSERT INTO device_enforcements (
        id, address_1, address_2, center_classification_date, city, classification, code_info, country, distribution_pattern, recall_event_id,
        initial_firm_notification, more_code_info, product_code, product_description, product_quantity, product_type, reason_for_recall,
        recall_initiation_date, recall_number, recalling_firm, report_date, state, status, termination_date, voluntary_mandated
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    cur.executemany(query, enforcement_values)
    conn.commit()

    cur.close()
    connection_pool.putconn(conn)

def device_event_import(file):
    """
    relevant files:
    device_event.sql
    device.sql
    patient.sql
    mdr_text.sql
    """
    with open(file, 'rb') as f:
        data = orjson.loads(f.read())

    conn = connection_pool.getconn()
    cur = conn.cursor()

    # Extract main device event data
    device_event_values = []
    devices = []
    patients = []
    patient_problems = []
    mdr = []
    for result in data['results']:
        event_id = str(uuid.uuid4())
        device_event_values.append((
            event_id,
            result.get("adverse_event_flag"),
            result.get("date_facility_aware"),
            result.get("date_manufacturer_received"),
            result.get("date_of_event"),
            result.get("date_received"),
            result.get("date_report"),
            result.get("date_report_to_fda"),
            result.get("date_report_to_manufacturer"),
            result.get("device_date_of_manufacturer"),
            result.get("distributor_address_1"),
            result.get("distributor_address_2"),
            result.get("distributor_city"),
            result.get("distributor_name"),
            result.get("distributor_state"),
            result.get("distributor_zip_code"),
            result.get("distributor_zip_code_ext"),
            result.get("event_key"),
            result.get("event_location"),
            result.get("event_type"),
            result.get("expiration_date_of_device"),
            result.get("health_professional"),
            result.get("initial_report_to_fda"),
            result.get("manufacturer_address_1"),
            result.get("manufacturer_address_2"),
            result.get("manufacturer_city"),
            result.get("manufacturer_contact_address_1"),
            result.get("manufacturer_contact_address_2"),
            result.get("manufacturer_contact_area_code"),
            result.get("manufacturer_contact_city"),
            result.get("manufacturer_contact_country"),
            result.get("manufacturer_contact_exchange"),
            result.get("manufacturer_contact_extension"),
            result.get("manufacturer_contact_f_name"),
            result.get("manufacturer_contact_l_name"),
            result.get("manufacturer_contact_pcity"),
            result.get("manufacturer_contact_pcountry"),
            result.get("manufacturer_contact_phone_number"),
            result.get("manufacturer_contact_plocal"),
            result.get("manufacturer_contact_postal_code"),
            result.get("manufacturer_contact_state"),
            result.get("manufacturer_contact_t_name"),
            result.get("manufacturer_contact_zip_code"),
            result.get("manufacturer_contact_zip_ext"),
            result.get("manufacturer_country"),
            result.get("manufacturer_g1_address_1"),
            result.get("manufacturer_g1_address_2"),
            result.get("manufacturer_g1_city"),
            result.get("manufacturer_g1_country"),
            result.get("manufacturer_g1_name"),
            result.get("manufacturer_g1_postal_code"),
            result.get("manufacturer_g1_state"),
            result.get("manufacturer_g1_zip_code"),
            result.get("manufacturer_g1_zip_code_ext"),
            result.get("manufacturer_link_flag"),
            result.get("manufacturer_name"),
            result.get("manufacturer_postal_code"),
            result.get("manufacturer_state"),
            result.get("manufacturer_zip_code"),
            result.get("manufacturer_zip_code_ext"),
            result.get("mdr_report_key"),
            result.get("number_devices_in_event"),
            result.get("number_patients_in_event"),
            result.get("previous_use_code"),
            result.get("product_problems"),
            result.get("product_problem_flag"),
            result.get("remedial_action"),
            result.get("removal_correction_number"),
            result.get("report_date"),
            result.get("report_number"),
            result.get("report_source_code"),
            result.get("report_to_fda"),
            result.get("report_to_manufacturer"),
            result.get("reporter_occupation_code"),
            result.get("reprocessed_and_reused_flag"),
            result.get("single_use_flag"),
            result.get("source_type"),
            result.get("type_of_report")
        ))
        for patient in result['patient']:
            patient_id = str(uuid.uuid4())
            patients.append((
                event_id,
                patient_id,       
                patient.get("date_received"), 
                patient.get("patient_sequence_number"),
                patient.get("patient_age"), 
                patient.get("patient_sex"), 
                patient.get("patient_weight"),
                patient.get("patient_ethnicity"), 
                patient.get("patient_race")
            ))
            patient_problems.append((
                event_id,
                patient_id,
                patient.get("patient_sequence_number"),
                patient.get("problem_description"),
                patient.get("outcome"),
                patient.get("treatment_description")
            ))
        
        for device in result['device']:
            devices.append((
                event_id,
                device.get("brand_name"), device.get("catalog_number"), device.get("date_received"),
                device.get("date_removed_flag"), device.get("date_returned_to_manufacturer"),
                device.get("device_age_text"), device.get("device_availability"),
                device.get("device_evaluated_by_manufacturer"), device_event_values[-1][0],
                device.get("device_operator"), device.get("device_report_product_code"),
                device.get("device_sequence_number"), device.get("expiration_date_of_device"),
                device.get("generic_name"), device.get("udi_di"), device.get("implant_flag"),
                device.get("lot_number"), device.get("manufacturer_d_address_1"),
                device.get("manufacturer_d_address_2"), device.get("manufacturer_d_city"),
                device.get("manufacturer_d_country"), device.get("manufacturer_d_name"),
                device.get("manufacturer_d_postal_code"), device.get("manufacturer_d_state"),
                device.get("manufacturer_d_zip_code"), device.get("manufacturer_d_zip_code_ext"),
                device.get("model_number"), device.get("other_id_number"), device.get("udi_public")
            ))

        for mdr_text in result['mdr_text']:
            mdr.append((
                event_id,
                mdr_text.get("mdr_text_key"),
                mdr_text.get("patient_sequence_number"),
                mdr_text.get("text").rstrip("\\"),
                mdr_text.get("text_type_code")
            ))

    #insert device events
    data_stream = io.StringIO()
    data_stream.writelines(
        "\t".join(map(format_value, row)) + "\n" for row in device_event_values
    )
    
    data_stream.seek(0)  # Reset stream position

    # Perform bulk insert using copy_from
    cur.copy_from(data_stream, 'device_event', sep="\t", null='\\N',
              columns=[
                  "event_id", "adverse_event_flag", "date_facility_aware", "date_manufacturer_received",
                  "date_of_event", "date_received", "date_report", "date_report_to_fda", "date_report_to_manufacturer",
                  "device_date_of_manufacturer", "distributor_address_1", "distributor_address_2", "distributor_city",
                  "distributor_name", "distributor_state", "distributor_zip_code", "distributor_zip_code_ext",
                  "event_key", "event_location", "event_type", "expiration_date_of_device", "health_professional",
                  "initial_report_to_fda", "manufacturer_address_1", "manufacturer_address_2", "manufacturer_city",
                  "manufacturer_contact_address_1", "manufacturer_contact_address_2", "manufacturer_contact_area_code",
                  "manufacturer_contact_city", "manufacturer_contact_country", "manufacturer_contact_exchange",
                  "manufacturer_contact_extension", "manufacturer_contact_f_name", "manufacturer_contact_l_name",
                  "manufacturer_contact_pcity", "manufacturer_contact_pcountry", "manufacturer_contact_phone_number",
                  "manufacturer_contact_plocal", "manufacturer_contact_postal_code", "manufacturer_contact_state",
                  "manufacturer_contact_t_name", "manufacturer_contact_zip_code", "manufacturer_contact_zip_ext",
                  "manufacturer_country", "manufacturer_g1_address_1", "manufacturer_g1_address_2",
                  "manufacturer_g1_city", "manufacturer_g1_country", "manufacturer_g1_name",
                  "manufacturer_g1_postal_code", "manufacturer_g1_state", "manufacturer_g1_zip_code",
                  "manufacturer_g1_zip_code_ext", "manufacturer_link_flag", "manufacturer_name",
                  "manufacturer_postal_code", "manufacturer_state", "manufacturer_zip_code",
                  "manufacturer_zip_code_ext", "mdr_report_key", "number_devices_in_event",
                  "number_patients_in_event", "previous_use_code", "product_problems",
                  "product_problem_flag", "remedial_action", "removal_correction_number",
                  "report_date", "report_number", "report_source_code", "report_to_fda",
                  "report_to_manufacturer", "reporter_occupation_code", "reprocessed_and_reused_flag",
                  "single_use_flag", "source_type", "type_of_report"
              ])
    #cur.executemany(queries.device_event_insert, device_event_values)
    conn.commit()

    #insert patient values
    #query = """
    #INSERT INTO patient (
    #    event_id, patient_id, date_received, patient_sequence_number, patient_age,
    #    patient_sex, patient_weight, patient_ethnicity, patient_race
    #) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    #"""
    #cur.executemany(query, patients)
    data_stream = io.StringIO()
    data_stream.writelines(
        "\t".join(map(format_value, row)) + "\n" for row in patients
    )
    data_stream.seek(0)

    cur.copy_from(data_stream, 'patient', sep="\t", null='\\N',columns = [
        "event_id",
        "patient_id",
        "date_received",
        "patient_sequence_number",
        "patient_age",
        "patient_sex",
        "patient_weight",
        "patient_ethnicity",
        "patient_race"
    ])
    conn.commit()

    #get patient problems
    #query = """
    #INSERT INTO patient_problem (event_id, patient_id, patient_sequence_number, problem_description)
    #VALUES (%s, %s, %s, %s)
    #"""
    #cur.executemany(query, patient_problems)
    data_stream = io.StringIO()
    data_stream.writelines(
        "\t".join(map(format_value, row)) + "\n" for row in patient_problems
    )
    data_stream.seek(0)

    cur.copy_from(data_stream, 'patient_problem', sep="\t", null='\\N',columns = [
        "event_id",
        "patient_id",
        "patient_sequence_number",
        "problem_description",
        "outcome",
        "treatment_description"
    ])
    conn.commit()

    
    #query = """
    #INSERT INTO device (
    #    event_id, brand_name, catalog_number, date_received, date_removed_flag, 
    #    date_returned_to_manufacturer, device_age_text, device_availability, 
    #    device_evaluated_by_manufacturer, device_event_key, device_operator, 
    #    device_report_product_code, device_sequence_number, expiration_date_of_device, 
    #    generic_name, udi_di, implant_flag, lot_number, manufacturer_d_address_1, 
    #    manufacturer_d_address_2, manufacturer_d_city, manufacturer_d_country, 
    #    manufacturer_d_name, manufacturer_d_postal_code, manufacturer_d_state, 
    #    manufacturer_d_zip_code, manufacturer_d_zip_code_ext, model_number, 
    #    other_id_number, udi_public
    #) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
    #"""
    #cur.executemany(query, devices)

    data_stream = io.StringIO()
    data_stream.writelines(
        "\t".join(map(format_value, row)) + "\n" for row in devices
    )

    data_stream.seek(0)

    cur.copy_from(data_stream, 'device', sep="\t", null='\\N',columns = [
        "event_id", "brand_name", "catalog_number", "date_received", "date_removed_flag", 
        "date_returned_to_manufacturer", "device_age_text", "device_availability", 
        "device_evaluated_by_manufacturer", "device_event_key", "device_operator", 
        "device_report_product_code", "device_sequence_number", "expiration_date_of_device", 
        "generic_name", "udi_di", "implant_flag", "lot_number", "manufacturer_d_address_1", 
        "manufacturer_d_address_2", "manufacturer_d_city", "manufacturer_d_country", 
        "manufacturer_d_name", "manufacturer_d_postal_code", "manufacturer_d_state", 
        "manufacturer_d_zip_code", "manufacturer_d_zip_code_ext", "model_number", 
        "other_id_number", "udi_public"
    ])
    conn.commit()

    #query = """
    #INSERT INTO mdr_text (event_id, mdr_text_key, patient_sequence_number, text, text_type_code)
    #VALUES (%s, %s, %s, %s, %s);
    #"""
    #cur.executemany(query, mdr)
    
    data_stream = io.StringIO()
    data_stream.writelines(
        "\t".join(map(format_value, row)) + "\n" for row in mdr
    )
    data_stream.seek(0)

    cur.copy_from(data_stream, 'mdr_text', sep="\t", null='\\N',columns = [
        "event_id",
        "mdr_text_key",
        "patient_sequence_number",
        "text",
        "text_type_code"
    ])
    conn.commit()

    cur.close()
    connection_pool.putconn(conn)

#skipping for now since the data doesn't seem very helpful 
def device_udi_import(file):
    """
    relevant files:
    deviceidentifier.sql
    """
    with open(file, 'r') as f:
        data = json.load(f)
    pass

def format_value(value):
    if value is None or value in ["", "\\"]:
        return "\\N"  # PostgreSQL NULL
    elif isinstance(value, list):
        return "{" + ",".join(f'"{v}"' for v in value) + "}"  # Format list as PostgreSQL array
    elif isinstance(value, str):
        return value.replace("\\", "\\\\").replace("\t", " ").replace("\n", " ").replace("\r","")
    return str(value).replace("\t", " ").replace("\n", " ")

def import_510k(file):
    """
    relevant files:
    510K.sql
    """
    with open(file, 'rb') as f:
        data = orjson.loads(f.read())
    
    conn = connection_pool.getconn()
    cur = conn.cursor()
    
    uuid_gen = (str(uuid.uuid4()) for _ in range(len(data['results'])))

    # Use a generator to process rows lazily and avoid large memory usage
    def generate_rows():
        for result in data['results']:
            yield (
                next(uuid_gen),
                result.get("address_1"),
                result.get("address_2"),
                result.get("city"),
                result.get("country_code"),
                result.get("postal_code"),
                result.get("state"),
                result.get("zip_code"),
                result.get("applicant"),
                result.get("contact"),
                result.get("advisory_committee_code"),
                result.get("advisory_committee_description"),
                result.get("clearance_type_code"),
                result.get("clearance_description"),
                result.get("date_received"),
                result.get("decision_code"),
                result.get("decision_date"),
                result.get("decision_description"),
                result.get("expedited_review_flag"),
                result.get("k_number"),
                result.get("product_code"),
                result.get("review_advisory_committee"),
                result.get("statement_or_summary"),
                result.get("third_party_flag"),
                *(result.get("openfda", {}).get(k) for k in [
                    "device_class",
                    "device_name",
                    "medical_specialty_description",
                    "fei_number",
                    "registration_number",
                    "regulation_number",
                ])
            )

    data_stream = io.StringIO()

    data_stream.writelines(
        "\t".join(map(format_value, row)) + "\n" for row in generate_rows()
    )


    data_stream.seek(0)

    cur.copy_from(data_stream, 'submission', sep="\t", null='\\N',columns = [
        "id",
        "address_1",
        "address_2",
        "city",
        "country_code",
        "postal_code",
        "state",
        "zip_code",
        "applicant",
        "contact",
        "advisory_committee_code",
        "advisory_committee_description",
        "clearance_type_code",
        "clearance_description",
        "date_received",
        "decision_code",
        "decision_date",
        "decision_description",
        "expedited_review_flag",
        "k_number",
        "product_code",
        "review_advisory_committee",
        "statement_or_summary",
        "third_party_flag",
        "device_class",
        "device_name",
        "medical_specialty_description",
        "fei_number",
        "registration_number",
        "regulation_number",
        
    ])
    conn.commit()

    cur.close()
    connection_pool.putconn(conn)

def device_pma_import(file):
    """
    relevant files:
    premarketapproval.sql
    """
    with open(file, 'rb') as f:
        data = orjson.loads(f.read())
    
    conn = connection_pool.getconn()
    cur = conn.cursor()

    pma_submission = [
        (
            str(uuid.uuid4()),
            result.get("advisory_committee_code"),
            result.get("advisory_committee_description"),
            result.get("ao_statement"),
            result.get("applicant"),
            result.get("city"),
            result.get("state"),
            result.get("street_1"),
            result.get("street_2"),
            result.get("zip"),
            result.get("zip_ext"),
            result.get("date_received"),
            result.get("decision_code"),
            result.get("decision_date"),
            result.get("docket_number"),
            result.get("expedited_review_flag"),
            result.get("fed_reg_notice_date"),
            result.get("generic_name"),
            result.get("pma_number"),
            result.get("product_code"),
            result.get("supplement_number"),
            result.get("supplement_reason"),
            result.get("supplement_type"),
            result.get("trade_name"),
            *(result.get("openfda", {}).get(k) for k in [
                "device_class",
                "device_name",
                "medical_specialty_description",
                "regulation_number",
                "fei_number",
                "registration_number",
            ])
        )
        for result in data['results']
    ]

    data_stream = io.StringIO()
    data_stream.writelines(
        "\t".join(map(format_value, row)) + "\n" for row in pma_submission
    )
    data_stream.seek(0)

    cur.copy_from(data_stream, 'pma_submission', sep="\t", null='\\N',columns = [
        "id",
        "advisory_committee_code",
        "advisory_committee_description",
        "ao_statement",
        "applicant",
        "city",
        "state",
        "street_1",
        "street_2",
        "zip",
        "zip_ext",
        "date_received",
        "decision_code",
        "decision_date",
        "docket_number",
        "expedited_review_flag",
        "fed_reg_notice_date",
        "generic_name",
        "pma_number",
        "product_code",
        "supplement_number",
        "supplement_reason",
        "supplement_type",
        "trade_name",
        "device_class",
        "device_name",
        "medical_specialty_description",
        "regulation_number",
        "registration_number",
        "fei_number",
    ])
    conn.commit()

    cur.close()
    connection_pool.putconn(conn)

def recall_import(file):
    """
    relevant files:
    recallsa.sql
    """
    with open(file, 'rb') as f:
        data = orjson.loads(f.read())
    
    conn = connection_pool.getconn()
    cur = conn.cursor()

    recalls = [
        (
            str(uuid.uuid4()),
            result.get("cfres_id"),
            result.get("k_number"),
            result.get("pma_number"),
            result.get("event_date_initiated"),
            result.get("event_date_created"),
            result.get("event_date_posted"),
            result.get("event_date_terminated"),
            result.get("recall_status"),
            result.get("recalling_firm"),
            result.get("firm_fei_number"),
            result.get("address_1"),
            result.get("address_2"),
            result.get("city"),
            result.get("state"),
            result.get("postal_code"),
            result.get("country"),
            result.get("additional_info_contact"),
            result.get("reason_for_recall"),
            result.get("other_submission_description"),
            result.get("product_description"),
            result.get("code_info"),
            result.get("product_code"),
            result.get("product_res_number"),
            result.get("product_quantity"),
            result.get("distribution_pattern"),
            result.get("res_event_number"),
            result.get("root_cause_description"),
            result.get("action"),
            *(result.get("openfda", {}).get(k) for k in [
                "device_class",
                "device_name",
                "medical_specialty_description",
                "regulation_number",
                "fei_number",
                "k_number",
                "registration_number",
            ])
        )
        for result in data['results']
    ]

    data_stream = io.StringIO()
    data_stream.writelines(
        "\t".join(map(format_value, row)) + "\n" for row in recalls
    )
    data_stream.seek(0)

    cur.copy_from(data_stream, 'recall', sep="\t", null='\\N',columns = [
        "id",
        "cfres_id",
        "k_number",
        "pma_number",
        "event_date_initiated",
        "event_date_created",
        "event_date_posted",
        "event_date_terminated",
        "recall_status",
        "recalling_firm",
        "firm_fei_number",
        "address_1",
        "address_2",
        "city",
        "state",
        "postal_code",
        "country",
        "additional_info_contact",
        "reason_for_recall",
        "other_submission_description",
        "product_description",
        "code_info",
        "product_code",
        "product_res_number",
        "product_quantity",
        "distribution_pattern",
        "res_event_number",
        "root_cause_description",
        "action",
        "device_class",
        "device_name",
        "medical_specialty_description",
        "regulation_number",
        "fei_number",
        "openfda_k_number",
        "registration_number"
    ])
    conn.commit()

    cur.close()
    connection_pool.putconn(conn)

def process_file(file):
    #print(f"Processing: {file}")
    if "device-event" in file:
        device_event_import(file)
    elif "device-enforcement" in file:
        device_enforcement_import(file)
    elif "device-classification" in file:
        device_classification_import(file)
    elif "device-udi" in file:
        return  # Skipping as per original code
    elif "510" in file:
        import_510k(file)
    elif "device-pma" in file:
        device_pma_import(file)
    elif "recall" in file:
        recall_import(file)
    else:
        print("File not recognized:", file)

def get_max_workers(memory_per_worker_gb=4):
    """Calculate the maximum number of workers based on free memory."""
    free_memory_gb = psutil.virtual_memory().available / (1024 ** 3)  # Convert bytes to GB
    return max(1, int(free_memory_gb / memory_per_worker_gb))  # Ensure at least 1 worker

# Ensure all connections are closed on exit
def close_pool():
    connection_pool.closeall()

if __name__ == '__main__':
    start_time = time.time() 

    conn = connection_pool.getconn()
    init_db.create_database()
    connection_pool.putconn(conn)

    import concurrent.futures
    atexit.register(close_pool)

    num_workers = get_max_workers()

    files = list_files_recursively(datapath)
    numFiles = len(files)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {executor.submit(process_file, file): file for file in files}

        for i, future in enumerate(concurrent.futures.as_completed(future_to_file), start=1):
            file = future_to_file[future]
            # print(f'start {i}/{numFiles} - {file}...')
            try:
                future.result()  # Retrieve any exceptions raised in the worker thread
            except Exception as e:
                print(f"Error processing {file}: {e}")

            print(f'{i}/{numFiles} - {file}...successful!')


    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
