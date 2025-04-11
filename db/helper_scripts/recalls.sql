CREATE TABLE recall (
    id UUID PRIMARY KEY,
    cfres_id TEXT, -- cfRes internal recall identifier
    k_number TEXT[],
    pma_number TEXT[],
    event_date_initiated DATE, -- Date firm first notified public
    event_date_created DATE, -- Date recall record created in FDA database
    event_date_posted DATE, -- Date FDA classified the recall
    event_date_terminated DATE, -- Date FDA terminated the recall
    recall_status TEXT, -- Current status of the recall
    recalling_firm TEXT, -- The firm responsible for the recall
    firm_fei_number TEXT, -- Facility identifier assigned by FDA
    address_1 TEXT, -- Address line 1 of the recalling firm
    address_2 TEXT, -- Address line 2 of the recalling firm
    city TEXT, -- City of recalling firm
    state TEXT, -- US state
    postal_code VARCHAR(20), -- ZIP or postal code
    country TEXT, -- Country of the recalling firm
    additional_info_contact TEXT, -- Contact info for further details
    reason_for_recall TEXT, -- Description of recall reason
    other_submission_description TEXT, -- Other regulatory descriptions
    product_description TEXT, -- Description of recalled product
    code_info TEXT, -- Lot/serial numbers, product codes
    product_code TEXT, -- FDA-assigned product code
    product_res_number TEXT, -- Product RES number
    product_quantity TEXT, -- Quantity of recalled product
    distribution_pattern TEXT, -- Distribution details
    res_event_number TEXT, -- FDA recall event number
    root_cause_description TEXT, -- General type of recall cause
    action TEXT, -- Actions taken as part of recall
    openfda_device_class TEXT,
    openfda_device_name TEXT,
    openfda_medical_specialty_description TEXT,
    openfda_regulation_number TEXT,
    openfda_fei_number TEXT[],
    openfda_k_number TEXT[],
    openfda_registration_number TEXT[]
);
