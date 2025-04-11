-- Assuming the following table structure exists (adjust based on your actual database design)

CREATE TABLE IF NOT EXISTS device_labels (
    device_label_id SERIAL PRIMARY KEY,
    brand_name VARCHAR(255),
    catalog_number VARCHAR(100),
    commercial_distribution_end_date DATE,
    commercial_distribution_status VARCHAR(50),
    company_name VARCHAR(255),
    customer_contacts TEXT,  -- Assuming a string for now - could be JSON or similar
    device_count_in_base_package INTEGER,
    device_description TEXT,
    device_sizes TEXT,
    has_donation_id_number BOOLEAN,
    is_combination_product Boolean,
    is_direct_marking_exempt Boolean,
    is_hct_p Boolean,
    is_sterilization_prior_use Boolean,
    sterilization_methods TEXT,  -- Assuming a comma-separated list
    public_version_date DATE,
    public_version_number VARCHAR(100),
    record_status BOOLEAN,
    record_key VARCHAR(255),
    labeler_duns_number VARCHAR(100) --  Consider using a more robust type like TEXT or STRING
);


-- Assuming the following table exists for Device Details:

CREATE TABLE IF NOT EXISTS device_details (
    device_detail_id SERIAL PRIMARY KEY,
    device_name VARCHAR(255),
    device_description TEXT,
    device_sizes TEXT,  -- Assuming a JSON array
    is_sterile BOOLEAN,
    sterilization_methods TEXT, --Assuming a comma-separated list.
    manufacturer_duns_number VARCHAR(100)
);

-- Assuming the following table exists for Device Type:

CREATE TABLE IF NOT EXISTS device_type (
    device_type_id SERIAL PRIMARY KEY,
    device_name VARCHAR(255),
    device_description TEXT,
    device_sizes TEXT, --Assuming a JSON array
    device_count_in_base_package INTEGER,
    device_number_of_devices INTEGER
);

-- Example Data Insert Statements (Illustrative - adjust as needed)

--INSERT DeviceLabels INTO DeviceLabels (brand_name VARCHAR(255), catalog_number VARCHAR(100), commercial_distribution_end_date DATE, commercial_distribution_status VARCHAR(50), company_name VARCHAR(255), customer_contacts TEXT, device_count_in_base_package INTEGER, device_description TEXT, device_sizes TEXT, has_donation_id_number BOOLEAN, is_combination_product Boolean, is_direct_marking_exempt Boolean, is_hct_p Boolean, is_sterilization_prior_use Boolean, sterilization_methods TEXT, public_version_date DATE, public_version_number VARCHAR(100), record_status BOOLEAN, device_count_in_base_package INTEGER, device_description TEXT, device_sizes TEXT, has_donation_id_number BOOLEAN, is_combination_product Boolean, is_direct_marking_exempt Boolean, is_hct_p Boolean, is_sterilization_prior_use Boolean, sterilization_methods TEXT, public_version_date DATE, public_version_number VARCHAR(100), record_status BOOLEAN, device_count_in_base_package INTEGER, device_description TEXT, device_sizes TEXT

--INSERT DeviceLabels INTO DeviceLabels (brand_name VARCHAR(255), catalog_number VARCHAR(100), commercial_distribution_end_date DATE, commercial_distribution_status VARCHAR(50), company_name VARCHAR(255), customer_contacts TEXT, device_count_in_base_package INTEGER, device_description TEXT, device_sizes TEXT, has_donation_id_number BOOLEAN, is_combination_product Boolean, is_direct_marking_exempt Boolean, is_hct_p Boolean, is_sterilization_prior_use Boolean, sterilization_methods TEXT, public_version_date DATE, public_version_number VARCHAR(100), record_status BOOLEAN, device_count_in_base_package INTEGER, device_description TEXT, device_sizes TEXT)
