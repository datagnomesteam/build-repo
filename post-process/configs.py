# DB credentials
db_credentials = {
    "username": "willbaldwin",
    "password": "password123",
    "database": "cse6242",
    "host": "localhost",
    "port": "5432"
}

# Sources
sources = ["device_events", "recall", "pma_submission", "submission"]

# Columns to pivot and 1 hot encode
pivot_cols = ["event_type", "device_class", "medical_specialty_description"]

# Master map of all canonicals, with binary for whether or not to clean
canonicals = {
    "source_id": 0,
    "device_name": 1, 
    "granular_name": 1, 
    "generic_name": 1, 
    "device_class": 0, 
    "manufacturer_name": 1,
    "manufacturer_fei": 0,
    "pma_number": 1,
    "k_number": 0,
    "event_type": 0,
    "event_status": 0,
    "event_date": 0,
    "implant_flag": 0,
    "manufacturer_street": 1,
    "manufacturer_city": 1,
    "manufacturer_state": 1,
    "manufacturer_postal_code": 1,
    "manufacturer_country": 1,
    "regulation_number": 0,
    "product_quantity": 0,
    "medical_specialty_description": 0
}
# Define mappings of canonical field: raw field
mappings = {
    "device_events": {
        "source_id": "event_id",
        "device_name": "openfda_device_name", 
        "granular_name": "brand_name", 
        "generic_name": "generic_name",
        "device_class": "openfda_device_class",
        "event_type": "event_type",
        "event_date": "date_of_event",
        "event_status": "remedial_action",
        "manufacturer_name": "manufacturer_d_name",
        "manufacturer_street": "manufacturer_d_address_1",
        "manufacturer_city": "manufacturer_d_city",
        "manufacturer_state": "manufacturer_d_state",
        "manufacturer_zip": "manufacturer_d_postal_code",
        "manufacturer_country": "manufacturer_d_country",
        "regulation_number": "openfda_regulation_number",
        "medical_specialty_description": "openfda_medical_specialty_description"
    },
    "submission": {
        "source_id": "id",
        "device_name": "openfda_device_name", 
        "device_class": "openfda_device_class",
        "manufacturer_name": "applicant",
        "k_number": "k_number",
        "event_status": "decision_code",
        "event_date": "decision_date",
        "manufacturer_street": "address_1",
        "manufacturer_city": "city",
        "manufacturer_state": "state",
        "manufacturer_zip": "postal_code",
        "manufacturer_country": "country_code",
        "regulation_number": "openfda_regulation_number",
        "medical_specialty_description": "openfda_medical_specialty_description"
    },
    "pma_submission": {
        "source_id": "id",
        "device_name": "openfda_device_name", 
        "granular_name": "trade_name", 
        "generic_name": "generic_name",
        "device_class": "openfda_device_class",
        "manufacturer_name": "applicant",
        "pma_number": "pma_number",
        "event_status": "decision_code",
        "event_date": "decision_date",
        "manufacturer_street": "street_1",
        "manufacturer_city": "city",
        "manufacturer_state": "state",
        "manufacturer_zip": "zip",
        "regulation_number": "openfda_regulation_number",
        "medical_specialty_description": "openfda_medical_specialty_description"
    },
    "recall": {
        "source_id": "id",
        "device_name": "openfda_device_name", 
        "device_class": "openfda_device_class", 
        "manufacturer_name": "recalling_firm",
        "manufacturer_fei": "firm_fei_number",
        "k_number": "openfda_k_number",
        "event_type": "source",
        "event_status": "recall_status",
        "event_date": "event_date_initiated",
        "manufacturer_street": "address_1",
        "manufacturer_city": "city",
        "manufacturer_state": "state",
        "manufacturer_zip": "postal_code",
        "manufacturer_country": "country",
        "regulation_number": "openfda_regulation_number",
        "product_quantity": "product_quantity",
        "medical_specialty_description": "openfda_medical_specialty_description"
    }
}
