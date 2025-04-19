"""
run this file with the command
    $ python -m streamlit run app.py
to open dashboard on port 8051

NOTE: please set your own password in this file and init_recall_db.py
"""

import streamlit as st
import pandas as pd
from psycopg2.extras import RealDictCursor
import plotly.express as px
from datetime import datetime, timedelta
from db_info import get_db_connection
from forecast import build_forecast_data, forecast, plot_timeseries
from datetime import date
from dateutil.relativedelta import relativedelta
import pydeck as pdk

sample_percent = 0.1

# connect to local db
def get_database_connection():
    try:
        conn = get_db_connection()
        return conn
    except Exception as e:
        st.error(f'Database connection error: {e}')
        return None

# function to fetch recall data; filter is dict {col: value}
#@st.cache_data
def fetch_events(filters=None):
    global sample_percent
    conn = get_database_connection()
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
        SELECT 
            d.openfda_device_name,
            de.event_id,
            de.adverse_event_flag,
            de.date_facility_aware,
            de.date_of_event,
            de.date_report_to_fda,
            de.date_report_to_manufacturer,
            de.event_type,
            de.health_professional,
            de.product_problems, 
            de.number_devices_in_event,
            de.number_patients_in_event,
            d.manufacturer_d_name,
            d.manufacturer_d_country,
            d.manufacturer_d_postal_code,
            d.manufacturer_d_state
        FROM sampled_events de
        JOIN device d ON d.event_id = de.event_id
        """

        where_clauses = []
        params = []

        if filters:
            if filters.get('openfda_device_name'):
                where_clauses.append('d.openfda_device_name ILIKE %s')
                params.append(f'%{filters["openfda_device_name"]}%')
            
            if filters.get('manufacturer_d_name'):
                where_clauses.append('d.manufacturer_d_name ILIKE %s')
                params.append(f'%{filters["manufacturer_d_name"]}%')
            
            if filters.get('date_from') and filters.get('date_to'):
                where_clauses.append('de.date_of_event BETWEEN %s AND %s')
                params.append(filters["date_from"])
                params.append(filters["date_to"])

            if filters.get('event_type'):
                where_clauses.append('de.event_type ILIKE %s')
                params.append(f'%{filters["event_type"]}%')


        if where_clauses:
            query += ' WHERE ' + ' AND '.join(where_clauses)
        
        query += ' ORDER BY de.date_of_event DESC'

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            results = cur.fetchall()

        df = pd.DataFrame(results)
        #df = df[pd.to_datetime(df["date_of_event"]) <= pd.Timestamp('2023-12-31')]

        return df
    
    except Exception as e:
        st.error(f'Error fetching data: {e}')
        return pd.DataFrame()
    finally:
        conn.close()

# app layout and functionality
def main():
    st.set_page_config(
        page_title="OpenFDA Medical Device - Adverse Events",
        page_icon="⚕️",
        layout="wide"
    )
    
    st.title("Adverse Events Dashboard")
    st.text("This dashboard takes a closer look at adverse events associated with medical devices. On the left sidebar, the user may apply a variety of filters on the underlying data. Visuals will update to reflect the filtered data.")
    st.text("At the top of the page, to the right of the sidebar, adverse events are summarized are broken down by type and manufacturer.")
    st.text("In the middle of the page, events are displayed over time, and the top 100 rows from the underlying event data is displayed in a tabular view. Holt-Winters exponential smoothing is implemented to forecast events beyond the specified time window. Major policy changes regarding medical devices are visualized along the X-axis.")
    st.text("At the bottom of the page, events are displayed geographically. This graph shows the locations of the manufacturers that produced the devices. The FDA data did not provide the address of events. To operate the map, you can identify hotspots with the default zoom level. From there you can zoom in using the slider to see a more granular view.")
    
    # sidebar filters
    st.sidebar.header("Filters")
    st.sidebar.text("Range of 3 years required to forecast...")

    # define the minimum and maximum selectable dates
    min_date = date(1945, 1, 1)
    max_date = date(2025, 1, 31)

    col1, col2 = st.sidebar.columns(2)

    with col1:
        # user selects the 'From' date
        date_from = st.date_input(
            "From",
            value=date(2000, 1, 1),
            min_value=min_date,
            max_value=max_date
        )

    with col2:
        # user selects the 'to' date with adjusted constraints
        date_to = st.date_input(
            "To",
            value=date(2024, 5, 31),
            min_value=min_date,
            max_value=max_date
        )
    
    # text filters
    device_name = st.sidebar.text_input("Device Name")
    recalling_firm = st.sidebar.text_input("Manufacturer")
    
    # event type filters
    event_type = st.sidebar.selectbox(
        "Adverse Event Type",
        ["All", "Other", "Death", "Malfunction", "Injury"]
    )
    
    # apply filters button
    if st.sidebar.button("Apply Filters"):
        filters = {
            'generic_name': device_name,
            'manufacturer_d_name': recalling_firm,
            'date_from': date_from,
            'date_to': date_to
        }
        
        if event_type != "All":
            filters['event_type'] = event_type
        
        df = fetch_events(filters)
    else:
        filters = {
            'date_from': date_from,
            'date_to': date_to
        }
        df = fetch_events(filters=filters)
    


    # display data and visualizations
    if not df.empty:
        # key metrics at the top
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        
        with metrics_col1:
            st.metric("Total Adverse Events", df.shape[0])

        with metrics_col3:
            recent_recalls = df[df['date_of_event'] >= (date_to - timedelta(days=30))].shape[0]
            st.metric("Events in Previous 30 Days", recent_recalls)
        
        # visualizations
        st.subheader("Event Statistics")
        
        col1, col2 = st.columns(2, gap='medium')
    
        with col1:
            # pie chart for recall root causes
            fig1 = px.pie(
                df, 
                names='event_type', 
                title='Events by Event Type',
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            fig1.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')

            st.plotly_chart(fig1)
        
        with col2:
            # bar chart for top manufacturers
            top_recalling_firm = df['manufacturer_d_name'].value_counts().head(10).reset_index()
            top_recalling_firm.columns = ['manufacturer_d_name', 'count']
            fig2 = px.bar(
                top_recalling_firm, 
                x='manufacturer_d_name', 
                y='count',
                title='Top 10 Manufacturers with Events'
            )
            st.plotly_chart(fig2)
        
        # time series chart
        monthly_counts = build_forecast_data(df=df, date_field='date_of_event', freq='ME')
        forecast_output = forecast(counts_df=monthly_counts, date_field='date_of_event', freq='ME')
        fig3 = plot_timeseries(df=forecast_output['df'], forecasted=forecast_output['forecasted'], rmse=forecast_output['rmse'], date_field='date_of_event', page='Events')
        st.plotly_chart(fig3)
        
        # data table with search and sort
        st.subheader("100 Most Revent Event Details")
        st.dataframe(
            df.drop(columns=['event_id']).sort_values('date_of_event', ascending=False)[:100],
            use_container_width=True
        )
        
        names = ["country code", "postal code", "place name", "admin name1", "admin code1", "admin name2", "admin code2", "admin name3", "admin code3", "latitude", "longitude", "accuracy"]
        zips = pd.read_csv('allCountries.txt',header=None, names=names,delimiter='\t')

        df['country code'] = df['manufacturer_d_country'].str.upper()
        df['postal code'] = df['manufacturer_d_postal_code'].astype(str).str.strip()
        zips['postal code'] = zips['postal code'].astype(str).str.strip()

        merged = pd.merge(
            df,
            zips[['country code', 'postal code', 'latitude', 'longitude']],
            on=['country code', 'postal code'],
            how='inner'
        )
        # Filter out rows with invalid lat/lon (e.g., NaN, 0, or out-of-bound values)
        merged = merged[(merged['latitude'].notna()) & 
                            (merged['longitude'].notna()) & 
                            (merged['latitude'] >= -90) & 
                            (merged['latitude'] <= 90) &
                            (merged['longitude'] >= -180) & 
                            (merged['longitude'] <= 180)]


        data = merged[['latitude', 'longitude']].rename(columns={
                'latitude': 'lat',
                'longitude': 'lon'
            })
        
        data = data.dropna(subset=['lat', 'lon']).reset_index(drop=True)

        


        # Let the user pick a zoom level via slider (simulated zoom control)
        zoom_level = st.slider("Simulated Zoom Level", min_value=6, max_value=16, value=9)

        # Function to compute hex radius based on zoom level
        def zoom_to_radius(zoom):
            return int(20000 / (2 ** (zoom - 8)))

        radius = zoom_to_radius(zoom_level)
        #st.write(f"🔍 Hex Radius: {radius} meters")

        # Define a custom color range with 10 intensity levels
        color_range = [
            [255, 255, 204],
            [255, 237, 160],
            [254, 217, 118],
            [254, 178, 76],
            [253, 141, 60],
            [252, 78, 42],
            [227, 26, 28],
            [189, 0, 38],
            [128, 0, 38],
            [77, 0, 75]
        ]

        # -- Create HexagonLayer --
        layer = pdk.Layer(
            'HexagonLayer',
            data,
            get_position='[lon, lat]',
            auto_highlight=True,
            elevation_scale=50,
            pickable=True,
            elevation_range=[0, 1000],
            extruded=True,
            coverage=1,
            radius=radius,
            color_aggregation='sum',
            color_range=color_range,
        )

        # --- View state ---
        view_state = pdk.ViewState(
            zoom=zoom_level,
            pitch=40,
            latitude=data['lat'].mean(),
            longitude=data['lon'].mean(),  
        )

        # --- Show map ---
        st.pydeck_chart(pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={"text": "Count: {elevationValue}"}
        ))


    else:
        st.info("No events found with the selected filters.")

if __name__ == "__main__":
    main()


