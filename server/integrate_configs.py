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
    "event_status": 1,
    "event_date": 0,
    "implant_flag": 0,
    "manufacturer_street": 1,
    "manufacturer_city": 1,
    "manufacturer_state": 1,
    "manufacturer_postal_code": 1,
    "manufacturer_country": 1,
    "regulation_number": 0,
    "product_quantity": 0,
    "medical_specialty_description": 0,
    "source": 0
}

# Define mappings of raw field: canonical field
mappings = {
    "device_events": {
        "event_id": "source_id",
        "openfda_device_name": "device_name",
        "brand_name": "granular_name",
        "generic_name": "generic_name",
        "openfda_device_class": "device_class",
        "event_type": "event_type",
        "date_of_event": "event_date",
        "remedial_action": "event_status",
        "manufacturer_d_name": "manufacturer_name",
        "manufacturer_d_address_1": "manufacturer_street",
        "manufacturer_d_city": "manufacturer_city",
        "manufacturer_d_state": "manufacturer_state",
        "manufacturer_d_postal_code": "manufacturer_zip",
        "manufacturer_d_country": "manufacturer_country",
        "openfda_regulation_number": "regulation_number",
        "openfda_medical_specialty_description": "medical_specialty_description"
    },
    "submission": {
        "id": "source_id",
        "openfda_device_name": "device_name",
        "openfda_device_class": "device_class",
        "applicant": "manufacturer_name",
        "k_number": "k_number",
        "decision_code": "event_status",
        "decision_date": "event_date",
        "address_1": "manufacturer_street",
        "city": "manufacturer_city",
        "state": "manufacturer_state",
        "postal_code": "manufacturer_zip",
        "country_code": "manufacturer_country",
        "openfda_regulation_number": "regulation_number",
        "openfda_medical_specialty_description": "medical_specialty_description"
    },
    "pma_submission": {
        "id": "source_id",
        "openfda_device_name": "device_name",
        "trade_name": "granular_name",
        "generic_name": "generic_name",
        "openfda_device_class": "device_class",
        "applicant": "manufacturer_name",
        "pma_number": "pma_number",
        "decision_code": "event_status",
        "decision_date": "event_date",
        "street_1": "manufacturer_street",
        "city": "manufacturer_city",
        "state": "manufacturer_state",
        "zip": "manufacturer_zip",
        "openfda_regulation_number": "regulation_number",
        "openfda_medical_specialty_description": "medical_specialty_description"
    },
    "recall": {
        "id": "source_id",
        "openfda_device_name": "device_name",
        "openfda_device_class": "device_class",
        "recalling_firm": "manufacturer_name",
        "firm_fei_number": "manufacturer_fei",
        "openfda_k_number": "k_number",
        "source": "event_type",
        "recall_status": "event_status",
        "event_date_initiated": "event_date",
        "address_1": "manufacturer_street",
        "city": "manufacturer_city",
        "state": "manufacturer_state",
        "postal_code": "manufacturer_zip",
        "country": "manufacturer_country",
        "openfda_regulation_number": "regulation_number",
        "product_quantity": "product_quantity",
        "openfda_medical_specialty_description": "medical_specialty_description"
    }
}
