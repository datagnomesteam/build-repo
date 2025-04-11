-- Create a table for storing data
CREATE TABLE
IF NOT EXISTS data_table
(
    id SERIAL PRIMARY KEY,
    name VARCHAR
(255) NOT NULL,
    value FLOAT,
    category VARCHAR
(100),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better query performance
CREATE INDEX
IF NOT EXISTS idx_data_category ON data_table
(category);
CREATE INDEX
IF NOT EXISTS idx_data_timestamp ON data_table
(timestamp);

-- Add some initial sample data
INSERT INTO data_table
    (name, value, category)
VALUES
    ('sample1', 10.5, 'category1');
INSERT INTO data_table
    (name, value, category)
VALUES
    ('sample2', 20.3, 'category1');
INSERT INTO data_table
    (name, value, category)
VALUES
    ('sample3', 15.7, 'category2');
INSERT INTO data_table
    (name, value, category)
VALUES
    ('sample4', 30.1, 'category2');
INSERT INTO data_table
    (name, value, category)
VALUES
    ('sample5', 25.9, 'category3'); 