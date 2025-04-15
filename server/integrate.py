import pandas as pd
import streamlit as st
from psycopg2.extras import RealDictCursor
import re
import uuid

def fetch_events(conn):
    conn = conn
    if not conn:
        return pd.DataFrame()
    
    # Step 1: Calculate the approximate row count from pg_class
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT reltuples::BIGINT AS total
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relname = 'device_event' AND n.nspname = 'public';
        """)
        total_rows = cur.fetchone()['total']

    # Step 2: Calculate the sample percentage (for approx 500k rows)
    target_count = 500000 #TODO: change back to 500k after testing
    sample_percent = min(100.0, max(0.01, (target_count / total_rows) * 100))
    sample_percent = round(sample_percent, 2)
    
    try:
        query = f"""
        WITH approx_count AS (
            SELECT reltuples::BIGINT AS total
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relname = 'device_event' AND n.nspname = 'public'
        ),
        sampled_events AS (
            SELECT * 
            FROM device_event TABLESAMPLE SYSTEM({sample_percent})
            REPEATABLE (200)
        )
        SELECT *
        FROM sampled_events de
        JOIN device d ON d.event_id = de.event_id
        """

        query += ' ORDER BY de.date_of_event DESC'

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            results = cur.fetchall()

        df = pd.DataFrame(results)
        return df
    
    except Exception as e:
        st.error(f'Error fetching data: {e}')
        return pd.DataFrame()
    
def fetch_table(conn, table):
    conn = conn
    if not conn:
        return pd.DataFrame()
    
    # Step 1: Calculate the approximate row count from pg_class
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(f"""
            SELECT reltuples::BIGINT AS total
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relname = '{table}' AND n.nspname = 'public';
        """)
        total_rows = cur.fetchone()['total']

    # Step 2: Calculate the sample percentage (for approx 500k rows)
    target_count = 500000 #TODO: change back to 500k after testing
    sample_percent = min(100.0, max(0.01, (target_count / total_rows) * 100))
    sample_percent = round(sample_percent, 2)
    
    try:
        query = f"""
        WITH approx_count AS (
            SELECT reltuples::BIGINT AS total
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relname = '{table}' AND n.nspname = 'public'
        ),
        sampled_table AS (
            SELECT * 
            FROM {table} TABLESAMPLE SYSTEM({sample_percent})
            REPEATABLE (200)
        )
        SELECT *
        FROM sampled_table
        """

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            results = cur.fetchall()

        df = pd.DataFrame(results)
        return df
    
    except Exception as e:
        st.error(f'Error fetching data: {e}')
        return pd.DataFrame()

def clean_canon(series):
    return (
        series.astype(str)
        .str.replace(r"[^a-zA-Z0-9]", " ", regex=True)
        .str.replace(r"\s{2,}", " ", regex=True)
        .str.strip()
        .str.lower()
        .replace("", pd.NA)
    )

def build_schema(df, mapping, canonicals):
    df = df.copy()
    print(df)
    # project raw columns to be mapped
    df = df[list(mapping.keys())]
    # rename columns based on mapping
    df = df.rename(columns=mapping)
    # ensure all canonical columns are present
    for canonical in canonicals:
        if canonical not in df.columns:
            df[canonical] = pd.NA
        else:
            df[canonical] = [','.join(map(str, l)) if isinstance(l, list) else l for l in df[canonical]]
            if canonicals[canonical] == 1:
                df[canonical] = clean_canon(df[canonical])
    # reorder columns to match canonical list
    df = df[list(canonicals.keys())]
    return df

@st.cache_data
def integrate(_conn, sources, pivot_cols, canonicals, mappings):

    # Read data from PostgreSQL database
    raw_tables = {}
    for source in sources:
        if source != "device_events":
            raw_tables[source] = fetch_table(_conn, source)
        else:
            # join events back onto devices to create device_events table
            raw_tables[source] = fetch_events(_conn)
    _conn.close()

    # Harmonize schemas
    tables_to_union = []
    for source in sources:
        mapping = mappings.get(source)
        df = raw_tables.get(source)
        df["source"] = source
        df = build_schema(df, mapping, canonicals)
        tables_to_union.append(df)

    # Merge
    master = pd.concat(tables_to_union, ignore_index=True)
    master.replace("", pd.NA, inplace=True)

    # Pivot and 1 hot encode
    for column in pivot_cols:
        dummies = pd.get_dummies(master[column])
        master = pd.concat([master, dummies], axis=1)

    return master