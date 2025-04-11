import streamlit as st
import pandas as pd
import sqlalchemy
from configs import db_credentials

# Database connection setup
db_connection_str = f"postgresql+psycopg2://{db_credentials['username']}:{db_credentials['password']}@{db_credentials['host']}:{db_credentials['port']}/{db_credentials['database']}"
db_engine = sqlalchemy.create_engine(db_connection_str)

# Function to load data from SQL view
def load_data(view_name):
    query = f'SELECT * FROM {view_name}'
    return pd.read_sql(query, con=db_engine)

# Load data
manufacturer_df = load_data('integrated_manufacturer_view')
device_df = load_data('integrated_device_view')
manufacturer_address_df = load_data('integrated_manufacturer_address_view')

# Streamlit app layout
st.title('Medical Devices Dashboard')

# Sidebar for search filter
search_query = st.sidebar.text_input('Search Manufacturers', '')

# Filter manufacturer data based on search query
if search_query:
    filtered_manufacturer_df = manufacturer_df[manufacturer_df['manufacturer_name'].str.contains(search_query, case=False, na=False)]
else:
    filtered_manufacturer_df = manufacturer_df

# Get the list of manufacturer names after filtering
filtered_manufacturer_names = filtered_manufacturer_df['manufacturer_name'].tolist()

# Filter devices and addresses based on the filtered manufacturers
filtered_device_df = device_df[device_df['manufacturer_name'].isin(filtered_manufacturer_names)]
filtered_address_df = manufacturer_address_df[manufacturer_address_df['manufacturer_name'].isin(filtered_manufacturer_names)]

# Display tables with sorting enabled
st.subheader('Manufacturers')
st.dataframe(filtered_manufacturer_df)

st.subheader('Devices')
st.dataframe(filtered_device_df)

st.subheader('Addresses')
st.dataframe(filtered_address_df)

# Cross-filtering: Update device and address tables based on selected manufacturer
selected_manufacturer = st.selectbox('Select Manufacturer', filtered_manufacturer_names)
if selected_manufacturer:
    manufacturer_name = manufacturer_df[manufacturer_df['manufacturer_name'] == selected_manufacturer]['manufacturer_name'].iloc[0]
    filtered_device_df = device_df[device_df['manufacturer_name'] == manufacturer_name]
    filtered_address_df = manufacturer_address_df[manufacturer_address_df['manufacturer_name'] == manufacturer_name]

    st.subheader(f'Devices for {selected_manufacturer}')
    st.dataframe(filtered_device_df)

    st.subheader(f'Addresses for {selected_manufacturer}')
    st.dataframe(filtered_address_df)