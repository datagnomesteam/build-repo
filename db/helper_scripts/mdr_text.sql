CREATE TABLE IF NOT EXISTS mdr_text (
    event_id UUID REFERENCES device_event(event_id) ON DELETE CASCADE, 
    mdr_text_key BIGINT PRIMARY KEY, -- Unique identifier for MDR text
    patient_sequence_number INT, -- Patient sequence number
    text TEXT, -- Narrative text or problem description
    text_type_code VARCHAR(1023) -- Describes the type of narrative
);