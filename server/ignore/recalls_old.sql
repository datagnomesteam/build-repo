CREATE TABLE IF NOT EXISTS recall (
    id SERIAL PRIMARY KEY,
    cfres_id VARCHAR(255), -- cfRes internal recall identifier
    event_date_initiated DATE, -- Date firm first notified public
    event_date_created DATE, -- Date recall record created in FDA database
    event_date_posted DATE, -- Date FDA classified the recall
    event_date_terminated DATE, -- Date FDA terminated the recall
    recall_status VARCHAR(255), -- Current status of the recall
    recalling_firm TEXT, -- The firm responsible for the recall
    firm_fei_number VARCHAR(255), -- Facility identifier assigned by FDA
    address_1 TEXT, -- Address line 1 of the recalling firm
    address_2 TEXT, -- Address line 2 of the recalling firm
    city VARCHAR(255), -- City of recalling firm
    state VARCHAR(100), -- US state
    postal_code VARCHAR(20), -- ZIP or postal code
    country VARCHAR(100), -- Country of the recalling firm
    additional_info_contact TEXT, -- Contact info for further details
    reason_for_recall TEXT, -- Description of recall reason
    other_submission_description TEXT, -- Other regulatory descriptions
    product_description TEXT, -- Description of recalled product
    code_info TEXT, -- Lot/serial numbers, product codes
    product_code VARCHAR(255), -- FDA-assigned product code
    product_res_number VARCHAR(255), -- Product RES number
    product_quantity VARCHAR(255), -- Quantity of recalled product
    distribution_pattern TEXT, -- Distribution details
    res_event_number VARCHAR(255), -- FDA recall event number
    root_cause_description TEXT, -- General type of recall cause
    action TEXT -- Actions taken as part of recall
);

CREATE TABLE IF NOT EXISTS recall_k_numbers (
    id SERIAL PRIMARY KEY,
    recall_id INT REFERENCES recall(id) ON DELETE CASCADE,
    k_number VARCHAR(255) -- FDA-assigned premarket notification number
);

CREATE TABLE IF NOT EXISTS recall_pma_numbers (
    id SERIAL PRIMARY KEY,
    recall_id INT REFERENCES recall(id) ON DELETE CASCADE,
    pma_number VARCHAR(255) -- FDA premarket application number
);

CREATE TABLE IF NOT EXISTS recall_openfda (
    id SERIAL PRIMARY KEY,
    recall_id INT REFERENCES recall(id) ON DELETE CASCADE,
    device_class VARCHAR(10) CHECK (device_class IN ('1', '2', '3', 'U', 'N', 'F')),
    device_name TEXT,
    medical_specialty_description TEXT
);

CREATE TABLE IF NOT EXISTS recall_openfda_fei_numbers (
    id SERIAL PRIMARY KEY,
    recall_openfda_id INT REFERENCES recall_openfda(id) ON DELETE CASCADE,
    fei_number VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS recall_openfda_k_numbers (
    id SERIAL PRIMARY KEY,
    recall_openfda_id INT REFERENCES recall_openfda(id) ON DELETE CASCADE,
    k_number VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS recall_openfda_registration_numbers (
    id SERIAL PRIMARY KEY,
    recall_openfda_id INT REFERENCES recall_openfda(id) ON DELETE CASCADE,
    registration_number VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS recall_openfda_regulation_numbers (
    id SERIAL PRIMARY KEY,
    recall_openfda_id INT REFERENCES recall_openfda(id) ON DELETE CASCADE,
    regulation_number VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS recall_meta (
    id SERIAL PRIMARY KEY,
    recall_id INT REFERENCES recall(id) ON DELETE CASCADE,
    disclaimer TEXT,
    license TEXT,
    last_updated DATE
);

CREATE TABLE IF NOT EXISTS recall_meta_results (
    id SERIAL PRIMARY KEY,
    recall_meta_id INT REFERENCES recall_meta(id) ON DELETE CASCADE,
    skip INT,
    limits INT,
    total INT
);
