CREATE TABLE IF NOT EXISTS patient (
    event_id UUID REFERENCES device_event(event_id) ON DELETE CASCADE, -- We will need to supply this ourselves to link the device events to the patients
    patient_id UUID PRIMARY KEY,
    date_received DATE,
    patient_sequence_number VARCHAR,
    patient_age VARCHAR,
    patient_sex VARCHAR,
    patient_weight VARCHAR,
    patient_ethnicity VARCHAR,
    patient_race VARCHAR
);

CREATE TABLE IF NOT EXISTS patient_problem (
    event_id UUID REFERENCES device_event(event_id) ON DELETE CASCADE,
    patient_id UUID PRIMARY KEY REFERENCES patient(patient_id) ON DELETE CASCADE,
    patient_sequence_number VARCHAR,
    problem_description TEXT,
    outcome VARCHAR CHECK (outcome IN ('Life Threatening', 'Hospitalization', 'Disability', 
                                        'Congenital Anomaly', 'Required Intervention', 
                                        'Other', 'Invalid Data', 'Unknown', 
                                        'No Information', 'Not Applicable', 'Death')),
    treatment_description TEXT
);