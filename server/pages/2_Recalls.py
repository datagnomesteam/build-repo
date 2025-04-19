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
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
from db_info import get_db_connection, get_db_cursor
from forecast import build_forecast_data, forecast, plot_timeseries

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

# app layout and functionality
def main():
    st.set_page_config(
        page_title="OpenFDA Medical Devices - Recalls",
        page_icon="⚕️",
        layout="wide"
    )
    
    st.title("Recalls Dashboard")
    st.text("This dashboard takes a closer look at medical device recalls. On the left sidebar, the user may apply a variety of filters on the underlying data. Visuals will update to reflect the filtered data.")
    st.text("At the top of the page, to the right of the sidebar, recalls are summarized are broken down by type and manufacturer.")
    st.text("In the middle of the page, recalls are displayed over time, and the top 100 rows from the underlying recall data is displayed in a tabular view. Holt-Winters exponential smoothing is implemented to forecast recalls beyond the specified time window. Major policy changes regarding medical devices are visualized along the X-axis.")
    st.text("At the bottom of the page, events are displayed geographically. This graph shows the locations of the manufacturers that produced the recalled devices. The FDA data did not provide the address of events. To operate the map, you can identify hotspots with the default zoom level. From there you can zoom in using the slider to see a more granular view.")
    
    # sidebar filters
    st.sidebar.header("Filters")
    st.sidebar.text("Range of 3 years required to forecast...")
    
    # date range filter
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
    recalling_firm = st.sidebar.text_input("Recalling Firm")
    
    # root cause filters
    root_cause_description = st.sidebar.selectbox(
        "Recall Root Cause",
        ["All", "Software design", "Use error", "Packaging", "Counterfeit"]
    )
    
    # apply filters button
    if st.sidebar.button("Apply Filters"):
        filters = {
            'device_name': device_name,
            'recalling_firm': recalling_firm,
            'date_from': date_from,
            'date_to': date_to
        }
        
        if root_cause_description != "All":
            filters['root_cause_description'] = root_cause_description
        
        df = fetch_recalls(filters)
    else:
        filters = {
            'date_from': min_date,
            'date_to': date_to
        }
        df = fetch_recalls()
    
    # display data and visualizations
    if not df.empty:
        # key metrics at the top
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        
        with metrics_col1:
            st.metric("Total Recalls", df.shape[0])

        with metrics_col3:
            recent_recalls = df[df['event_date_posted'] >= (date_to - timedelta(days=30))].shape[0]
            st.metric("Recalls in Previous 30 Days", recent_recalls)
        
        # visualizations
        st.subheader("Recall Statistics")
        
        col1, col2 = st.columns(2, gap='medium')
    
        with col1:
            # pie chart for recall device_class
            fig1 = px.pie(
                df, 
                names='openfda_device_class', 
                title='Recalls by Device Class',
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            fig1.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')

            st.plotly_chart(fig1)
        
        with col2:
            # bar chart for top recalling_firm
            top_recalling_firm = df['recalling_firm'].value_counts().head(10).reset_index()
            top_recalling_firm.columns = ['recalling_firm', 'count']
            fig2 = px.bar(
                top_recalling_firm, 
                x='recalling_firm', 
                y='count',
                title='Top 10 Firms with Recalls'
            )
            st.plotly_chart(fig2)
        
        # time series chart
        monthly_counts = build_forecast_data(df=df, date_field='event_date_posted', freq='ME')
        forecast_output = forecast(counts_df=monthly_counts, date_field='event_date_posted', freq='ME')
        fig3 = plot_timeseries(df=forecast_output['df'], forecasted=forecast_output['forecasted'], rmse=forecast_output['rmse'], date_field='event_date_posted', page='Recalls')
        st.plotly_chart(fig3)
        
        # data table with search and sort
        st.subheader("100 Most Recent Recall Details")
        st.dataframe(
            df.drop(columns=['id']).sort_values('event_date_posted', ascending=False)[:100],
            use_container_width=True
        )
        
    else:
        st.info("No recalls found with the selected filters.")

if __name__ == "__main__":
    main()


