CREATE TABLE submission (
    id UUID PRIMARY KEY,
    address_1 TEXT,
    address_2 TEXT,
    city VARCHAR(255),
    country_code VARCHAR(2),
    postal_code VARCHAR(20),
    state VARCHAR(50),
    zip_code VARCHAR(20),
    applicant VARCHAR(255),
    contact VARCHAR(255),
    advisory_committee_code VARCHAR(2),
    advisory_committee_description VARCHAR(255),
    clearance_type_code VARCHAR(20),
    clearance_description VARCHAR(255),
    date_received DATE,
    decision_code VARCHAR(4),
    decision_date DATE,
    decision_description TEXT,
    expedited_review_flag BOOLEAN,
    k_number VARCHAR(50) UNIQUE,
    product_code VARCHAR(50),
    review_advisory_committee VARCHAR(255),
    statement_or_summary TEXT,
    third_party_flag BOOLEAN,
    openfda_device_class VARCHAR(1),
    openfda_device_name VARCHAR(255),
    openfda_medical_specialty_description VARCHAR(255),
    openfda_fei_number TEXT[],
    openfda_registration_number TEXT[],
    openfda_regulation_number TEXT
);
