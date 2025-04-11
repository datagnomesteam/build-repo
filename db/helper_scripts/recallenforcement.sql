CREATE TABLE device_enforcements (
    id UUID PRIMARY KEY,
    address_1 TEXT,
    address_2 TEXT,
    center_classification_date DATE,
    city TEXT,
    classification TEXT,
    code_info TEXT,
    country TEXT,
    distribution_pattern TEXT,
    recall_event_id BIGINT,
    initial_firm_notification TEXT,
    more_code_info TEXT,
    product_code TEXT,
    product_description TEXT,
    product_quantity TEXT,
    product_type TEXT,
    reason_for_recall TEXT,
    recall_initiation_date DATE,
    recall_number TEXT,
    recalling_firm TEXT,
    report_date DATE,
    state TEXT,
    status TEXT,
    termination_date DATE,
    voluntary_mandated TEXT
);

--CREATE TABLE openfda (
--    id SERIAL PRIMARY KEY,
--    recall_id INT REFERENCES recalls(id),
--    is_original_packager BOOLEAN
--);

--CREATE TABLE openfda_details (
--    id SERIAL PRIMARY KEY,
--    openfda_id INT REFERENCES openfda(id),
--    key TEXT,
--    value TEXT
--);

--CREATE TABLE meta (
--    id SERIAL PRIMARY KEY,
--    disclaimer TEXT,
--    license TEXT,
--    last_updated DATE
--);

--CREATE TABLE results (
--    id SERIAL PRIMARY KEY,
--    meta_id INT REFERENCES meta(id),
--    skip BIGINT,
--    limit BIGINT,
--    total BIGINT
--);