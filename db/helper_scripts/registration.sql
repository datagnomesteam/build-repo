CREATE TABLE establishment_type (
    id SERIAL PRIMARY KEY,
    type VARCHAR(255) UNIQUE NOT NULL
);

CREATE TABLE registration (
    id SERIAL PRIMARY KEY,
    address_line_1 TEXT,
    address_line_2 TEXT,
    city VARCHAR(255),
    fei_number VARCHAR(50),
    initial_importer_flag BOOLEAN,
    iso_country_code VARCHAR(10),
    name VARCHAR(255),
    postal_code VARCHAR(20),
    reg_expiry_date_year INT,
    registration_number VARCHAR(50),
    state_code VARCHAR(50),
    status_code INT CHECK (status_code IN (1, 5)),
    zip_code VARCHAR(20)
);

CREATE TABLE owner_operator (
    id SERIAL PRIMARY KEY,
    firm_name VARCHAR(255),
    owner_operator_number VARCHAR(50)
);

CREATE TABLE contact_address (
    id SERIAL PRIMARY KEY,
    owner_operator_id INT REFERENCES owner_operator(id) ON DELETE CASCADE,
    address_1 TEXT,
    address_2 TEXT,
    city VARCHAR(255),
    iso_country_code VARCHAR(10),
    postal_code VARCHAR(20),
    state_code VARCHAR(50),
    state_province VARCHAR(50)
);

CREATE TABLE official_correspondent (
    id SERIAL PRIMARY KEY,
    owner_operator_id INT REFERENCES owner_operator(id) ON DELETE CASCADE,
    first_name VARCHAR(255),
    last_name VARCHAR(255),
    middle_initial VARCHAR(10),
    phone_number VARCHAR(50),
    subaccount_company_name VARCHAR(255)
);

CREATE TABLE us_agent (
    id SERIAL PRIMARY KEY,
    registration_id INT REFERENCES registration(id) ON DELETE CASCADE,
    address_line_1 TEXT,
    address_line_2 TEXT,
    bus_phone_area_code VARCHAR(10),
    bus_phone_extn VARCHAR(10),
    bus_phone_num VARCHAR(50),
    business_name VARCHAR(255),
    city VARCHAR(255),
    email_address VARCHAR(255),
    fax_area_code VARCHAR(10),
    fax_num VARCHAR(50),
    iso_country_code VARCHAR(10),
    name VARCHAR(255),
    postal_code VARCHAR(20),
    state_code VARCHAR(50),
    zip_code VARCHAR(20)
);

CREATE TABLE k_number (
    id SERIAL PRIMARY KEY,
    number VARCHAR(50) UNIQUE NOT NULL
);

CREATE TABLE pma_number (
    id SERIAL PRIMARY KEY,
    number VARCHAR(50) UNIQUE NOT NULL
);

CREATE TABLE proprietary_name (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL
);

CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    created_date DATE,
    exempt BOOLEAN,
    owner_operator_number VARCHAR(50),
    product_code VARCHAR(50)
);

CREATE TABLE openfda (
    id SERIAL PRIMARY KEY,
    product_id INT REFERENCES products(id) ON DELETE CASCADE,
    device_class VARCHAR(10) CHECK (device_class IN ('1', '2', '3', 'U', 'N', 'F')),
    device_name VARCHAR(255),
    medical_specialty_description VARCHAR(255),
    regulation_number VARCHAR(50)
);

-- Junction tables to handle one-to-many relationships
CREATE TABLE registration_establishment (
    registration_id INT REFERENCES registration(id) ON DELETE CASCADE,
    establishment_id INT REFERENCES establishment_type(id) ON DELETE CASCADE,
    PRIMARY KEY (registration_id, establishment_id)
);

CREATE TABLE registration_proprietary_name (
    registration_id INT REFERENCES registration(id) ON DELETE CASCADE,
    proprietary_name_id INT REFERENCES proprietary_name(id) ON DELETE CASCADE,
    PRIMARY KEY (registration_id, proprietary_name_id)
);

CREATE TABLE registration_k_number (
    registration_id INT REFERENCES registration(id) ON DELETE CASCADE,
    k_number_id INT REFERENCES k_number(id) ON DELETE CASCADE,
    PRIMARY KEY (registration_id, k_number_id)
);

CREATE TABLE registration_pma_number (
    registration_id INT REFERENCES registration(id) ON DELETE CASCADE,
    pma_number_id INT REFERENCES pma_number(id) ON DELETE CASCADE,
    PRIMARY KEY (registration_id, pma_number_id)
);
