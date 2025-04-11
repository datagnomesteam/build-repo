
CREATE TABLE medical_specialty (
    code CHAR(2) PRIMARY KEY,
    description VARCHAR(255) NOT NULL
);

CREATE TABLE device_classification (
    id UUID PRIMARY KEY,
    definition TEXT,
    device_class TEXT,
    device_name TEXT,
    gmp_exempt_flag TEXT,
    implant_flag TEXT,
    life_sustain_support_flag TEXT,
    medical_specialty_code CHAR(2),
    product_code VARCHAR(255),
    regulation_number VARCHAR(255),
    review_code VARCHAR(255),
    review_panel VARCHAR(255),
    submission_type_id TEXT,
    summary_malfunction_reporting TEXT,
    third_party_flag TEXT,
    unclassified_reason TEXT,
    openfda_pma_number TEXT[],
    openfda_fei_number TEXT[],
    openfda_k_number TEXT[],
    openfda_registration_number TEXT[]
);

INSERT INTO medical_specialty (code, description) VALUES
('AN', 'Anesthesiology'),
('CV', 'Cardiovascular'),
('CH', 'Clinical Chemistry'),
('DE', 'Dental'),
('EN', 'Ear, Nose, Throat'),
('GU', 'Gastroenterology, Urology'),
('HO', 'General Hospital'),
('HE', 'Hematology'),
('IM', 'Immunology'),
('MG', 'Medical Genetics'),
('MI', 'Microbiology'),
('NE', 'Neurology'),
('OB', 'Obstetrics/Gynecology'),
('OP', 'Ophthalmic'),
('OR', 'Orthopedic'),
('PA', 'Pathology'),
('PM', 'Physical Medicine'),
('RA', 'Radiology'),
('SU', 'General, Plastic Surgery'),
('TX', 'Clinical Toxicology');