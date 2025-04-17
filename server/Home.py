"""
run this file with the command
    $ python -m streamlit run home.py
to open dashboard on port 8051

-- have a pages directory to allow multiple links

NOTE: please set your own password in this file and init_recall_db.py
"""

import streamlit as st
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import plotly.express as px
from datetime import datetime, timedelta
from db_info import get_db_connection, get_db_cursor
from integrate import integrate
from integrate_configs import sources, pivot_cols, canonicals, mappings
import gc



# connect to local db
def get_database_connection():
    try:
        conn = get_db_connection()
        return conn
    except Exception as e:
        st.error(f'Database connection error: {e}')
        return None

# function to fetch recall data; filter is dict {col: value}
def fetch_recalls(filters=None):
    conn = get_database_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        query = """
        SELECT *
        FROM recall 
        """
        
        where_clauses = []
        params = []
        
        if filters:
            if filters.get('device_name'):
                where_clauses.append('device_name ILIKE %s')
                params.append(f'%{filters["device_name"]}%')
            
            if filters.get('recalling_firm'):
                where_clauses.append('recalling_firm ILIKE %s')
                params.append(f'%{filters["recalling_firm"]}%')
            
            if filters.get('date_from') and filters.get('date_to'):
                where_clauses.append('event_date_posted BETWEEN %s AND %s')
                params.append(filters["date_from"])
                params.append(filters["date_to"])
        
        if where_clauses:
            query += ' WHERE ' + ' AND '.join(where_clauses)
        
        query += ' ORDER BY event_date_posted DESC'
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            results = cur.fetchall()
        
        df =  pd.DataFrame(results)

        return df
    
    except Exception as e:
        st.error(f'Error fetching data: {e}')
        return pd.DataFrame()
    finally:
        conn.close()

# get statistics
def get_recall_stats():
    conn = get_database_connection()
    if not conn:
        return {}
    
    try:
        stats = {}

        # get recall counts by root cause
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT root_cause_description, COUNT(*) as count 
                FROM recall 
                GROUP BY root_cause_description 
                ORDER BY root_cause_description
            """)
            stats['class_counts'] = cur.fetchall()
               
        # top recalling_firm
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT recalling_firm, COUNT(*) as count 
                FROM device_recalls 
                GROUP BY recalling_firm 
                ORDER BY count DESC 
                LIMIT 10
            """)
            stats['top_recalling_firm'] = cur.fetchall()
        
        # recent trend (last 12 months)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT 
                    DATE_TRUNC('month', event_date_posted) as month, 
                    COUNT(*) as count 
                FROM device_recalls 
                WHERE event_date_posted >= NOW() - INTERVAL '1 year' 
                GROUP BY month 
                ORDER BY month
            """)
            stats['monthly_trend'] = cur.fetchall()
            
        return stats
    
    except Exception as e:
        st.error(f"Error fetching stats: {e}")
        return {}
    finally:
        conn.close()

def analyze_all_tables(conn):
    # Query to get all table names in the public schema
    query = """
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'
    """
    
    # Fetch the list of tables
    with conn.cursor() as cursor:
        cursor.execute(query)
        tables = cursor.fetchall()
    
    # Run ANALYZE for each table
    for table in tables:
        table_name = table[0]
        try:
            with conn.cursor() as cursor:
                #st.write(f"Running ANALYZE on {table_name}...")
                cursor.execute(f"ANALYZE public.{table_name}")
                conn.commit()  # Commit after each analyze
                #st.write(f"ANALYZE completed for {table_name}")
        except Exception as e:
            st.error(f"Error analyzing {table_name}: {str(e)}")


# app layout and functionality
def main():
    st.set_page_config(
        page_title="OpenFDA Medical Device - Home",
        page_icon="⚕️",
        layout="wide"
    )
    
    # Streamlit app layout
    st.title('OpenFDA Medical Devices Dashboard')

    # Sidebar for search filter
    search_query = st.sidebar.text_input('Search Manufacturers', '')

    #print(type(db_credentials))
    conn = get_database_connection()
    analyze_all_tables(conn)
    integrated = integrate(conn, sources, pivot_cols, canonicals, mappings)
    device_df = (
        integrated
            .groupby([
                "device_name", 
                "manufacturer_name", 
                "device_class", 
                "regulation_number", 
                "medical_specialty_description"
            ], dropna=False)
            .agg(
                deaths=("death", "sum"),
                injuries=("injury", "sum"),
                malfunctions=("malfunction", "sum"),
                other=("other", "sum"),
                recalls=("recall", "sum"),
                class_1=("1", "nunique"),
                class_2=("2", "nunique"),
                class_3=("3", "nunique"),
                records=("device_name", "count"),
                pma_submissions=("pma_number", "nunique"),
                k_submissions=("k_number", "nunique"),
            )
            .reset_index()
            .sort_values(by=["manufacturer_name", "device_class", "device_name"])
    )
    manufacturer_df = (
        integrated
            .groupby("manufacturer_name", dropna=False)
            .agg(
                deaths=("death", "sum"),
                injuries=("injury", "sum"),
                malfunctions=("malfunction", "sum"),
                other=("other", "sum"),
                recalls=("recall", "sum"),
                class_1=("1", "nunique"),
                class_2=("2", "nunique"),
                class_3=("3", "nunique"),
                unique_devices=("device_name", "nunique"),
                pma_submissions=("pma_number", "nunique"),
                k_submissions=("k_number", "nunique"),
            )
            .reset_index()
            .sort_values(by="manufacturer_name")
    )
    manufacturer_address_df = (
        integrated[[
            "manufacturer_name", 
            "manufacturer_street", 
            "manufacturer_city", 
            "manufacturer_state", 
            "manufacturer_country", 
            "manufacturer_postal_code"
        ]]
        .drop_duplicates()
        .sort_values(by="manufacturer_name")
        .reset_index(drop=True)
    )   
    
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

if __name__ == "__main__":
    main()


