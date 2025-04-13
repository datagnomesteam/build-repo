"""
run this file with the command
    $ python -m streamlit run app.py
to open dashboard on port 8051

NOTE: please set your own password in this file and init_recall_db.py
"""

import streamlit as st
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import plotly.express as px
from datetime import datetime, timedelta
from db_info import get_db_connection, get_db_cursor
from datetime import date
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go
from dateutil.relativedelta import relativedelta

sample_percent = 0.1

# connect to local db
def get_database_connection():
    try:
        conn = get_db_connection()
        return conn
    except Exception as e:
        st.error(f'Database connection error: {e}')
        return None
    
def forecast(monthly_counts):
    # fit the Exponential Smoothing Model
    try:
        model = ExponentialSmoothing(monthly_counts['count'], trend='add', seasonal='add', seasonal_periods=12)
    except ValueError:
        combined_df = monthly_counts.reset_index()
        combined_df['type'] = ['Actual'] * len(monthly_counts)
        return combined_df, False
    else:
        fit = model.fit()

        # generate Forecasts
        forecast_periods = 24  # forecasting 24 months ahead
        forecast_index = pd.date_range(start=monthly_counts.index[-1] + pd.DateOffset(months=1), periods=forecast_periods, freq='M')
        forecast = fit.forecast(forecast_periods)
        forecast_series = pd.Series(forecast, index=forecast_index)

        # combine actual and forecasted Data
        combined_series = pd.concat([monthly_counts['count'], forecast_series])
        combined_df = combined_series.reset_index()
        combined_df.columns = ['date_of_event', 'count']
        combined_df['type'] = ['Actual'] * len(monthly_counts) + ['Forecast'] * forecast_periods
        return combined_df, True

def plot_timeseries(df, forecasted):
    policy_changes = [
        {'date': '1906-01-01', 'label': 'Pure Food and Drugs Act'},
        {'date': '1938-01-01', 'label': 'Federal Food, Drug, and Cosmetic Act'},
        {'date': '1944-01-01', 'label': 'Public Health Service Act'},
        {'date': '1968-01-01', 'label': 'Radiation Control for Health and Safety Act'},
        {'date': '1976-01-01', 'label': 'Medical Device Amendments to the FD&C Act'},
        {'date': '1990-01-01', 'label': 'Safe Medical Devices Act'},
        {'date': '1992-01-01', 'label': 'Mammography Quality Standards Act'},
        {'date': '1997-01-01', 'label': 'FDA Modernization Act'},
        {'date': '2002-01-01', 'label': 'Medical Device User Fee and Modernization Act'},
        {'date': '2007-01-01', 'label': 'FDA Amendments Act'},
        {'date': '2012-01-01', 'label': 'FDA Safety and Innovation Act'},
        {'date': '2016-01-01', 'label': '21st Century Cures Act'},
        {'date': '2017-01-01', 'label': 'FDA Reauthorization Act'},
        {'date': '2020-01-01', 'label': 'Coronavirus Aid, Relief, and Economic Security Act'},
        {'date': '2022-01-01', 'label': 'FDAUFRA; FDORA; PREVENT Pandemics Act'}
    ]

    # Filter policy changes to include only those within the data range
    valid_policy_changes = [
        policy for policy in policy_changes
        if df['date_of_event'].min() <= pd.to_datetime(policy['date']) <= df['date_of_event'].max()
    ]
    
    # plot with plotly
    fig = go.Figure()

    # plot actual data
    fig.add_trace(go.Scatter(
        x=df[df['type'] == 'Actual']['date_of_event'],
        y=df[df['type'] == 'Actual']['count'],
        mode='lines',
        name='Actual',
        line=dict(color='blue')
    ))

    if forecasted:
        # plot forecasted data
        fig.add_trace(go.Scatter(
            x=df[df['type'] == 'Forecast']['date_of_event'],
            y=df[df['type'] == 'Forecast']['count'],
            mode='lines',
            name='Forecast',
            line=dict(color='red', dash='dash')
        ))
    
    # add vertical lines for each valid policy change
    for policy in valid_policy_changes:
        fig.add_vline(
            x=pd.to_datetime(policy['date']),
            line_width=2,
            line_dash="solid",
            line_color="black"
        )
        fig.add_annotation(
            x=pd.to_datetime(policy['date']),
            y=df['count'].max(),
            text=policy['label'],
            showarrow=False,
            font=dict(size=10, color="black"),
            align="center"
        )

    # add a hyperlinked footer
    fig.add_annotation(
        x=-10,
        y=0,
        xref="paper",
        yref="paper",
        text="<a href='https://www.fda.gov/medical-devices/overview-device-regulation/history-medical-device-regulation-oversight-united-states' target='_blank'>Read about Medical Device Policy</a>",
        showarrow=False,
        font=dict(size=12, color="blue"),
        align="center",
        xanchor="center",
        yanchor="top"
    )

    fig.update_layout(
        title='Events Over Time',
        xaxis_title='Date',
        yaxis_title='Number of Events',
        legend=dict(x=0, y=1),
        hovermode='x unified'
    )

    return fig

# function to fetch recall data; filter is dict {col: value}
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
    target_count = 500000
    sample_percent = min(100.0, max(0.01, (target_count*4 / total_rows) * 100))
    sample_percent = round(sample_percent, 2)
    st.write(sample_percent)
    
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

        return df
    
    except Exception as e:
        st.error(f'Error fetching data: {e}')
        return pd.DataFrame()
    finally:
        conn.close()

# get statistics
def get_event_stats():
    conn = get_database_connection()
    if not conn:
        return {}
    
    try:
        stats = {}

        # get event counts by type
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT event_type, COUNT(*) as count 
                FROM device_event 
                GROUP BY event_type 
                ORDER BY event_type
            """)
            stats['class_counts'] = cur.fetchall()
               
        # top manufacturer
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT manufacturer_d_name, COUNT(*) as count 
                FROM device 
                GROUP BY manufacturer_d_name 
                ORDER BY count DESC 
                LIMIT 10
            """)
            stats['top_recalling_firm'] = cur.fetchall()
        
        # recent trend (last 12 months)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT 
                    DATE_TRUNC('month', date_of_event) as month, 
                    COUNT(*) as count 
                FROM device_event 
                WHERE date_of_event >= NOW() - INTERVAL '1 year' 
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

# app layout and functionality
def main():
    st.set_page_config(
        page_title="OpenFDA Medical Device - Adverse Events",
        page_icon="⚕️",
        layout="wide"
    )
    
    st.title("Adverse Events Dashboard")
    
    # sidebar filters
    st.sidebar.header("Filters")

    # define the minimum and maximum selectable dates
    min_date = date(1945, 1, 1)
    max_date = date(2025, 1, 1)

    col1, col2 = st.sidebar.columns(2)

    with col1:
        # user selects the 'From' date
        date_from = st.date_input(
            "From",
            value=max_date - timedelta(days=365*3),
            min_value=min_date,
            max_value=max_date - timedelta(days=365*3)
        )

    # calculate the minimum 'To' date (2 years after 'From' date)
    min_to_date = max_date
    if date_from + relativedelta(years=2) <= max_date:
        min_to_date = date_from + relativedelta(years=2)

    with col2:
        # user selects the 'to' date with adjusted constraints
        date_to = st.date_input(
            "To",
            value=min_to_date,
            min_value=min_to_date,
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
        df = fetch_events()
    
    # display data and visualizations
    if not df.empty:
        # key metrics at the top
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        
        with metrics_col1:
            st.metric("Total Adverse Events", df.shape[0])
        
        #with metrics_col2:
        #    st.metric("Percentage of Events processed",sample_percent)
                
        # with metrics_col2:
        #     class_counts = df['root_cause_description'].value_counts()
        #     class_i_count = class_counts.get('Class I', 0)
        #     st.metric("Class I Recalls (Highest Risk)", class_i_count)

            
        with metrics_col3:
            recent_recalls = df[df['date_of_event'] >= (datetime.date(datetime.now()) - timedelta(days=30))].shape[0]
            st.metric("Events in Last 30 Days", recent_recalls)
        
        # visualizations
        st.subheader("Event Statistics")
        
        col1, col2 = st.columns(2, gap='medium')
    
        with col1:
            # pie chart for recall root causes
            fig1 = px.pie(
                df, 
                names='event_type', 
                title='Recalls by Event Type',
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
        df['date_of_event'] = pd.to_datetime(df['date_of_event'], errors='coerce')
        df = df[df['date_of_event'] >= pd.Timestamp('1902-01-01')]
        monthly_counts = df.groupby(pd.Grouper(key='date_of_event', freq='M'), dropna=True).size().reset_index(name='count')
        monthly_counts = monthly_counts.sort_values('date_of_event')
        monthly_counts.set_index('date_of_event', inplace=True)
        monthly_counts.index = pd.to_datetime(monthly_counts.index)

        #combined_data, success = forecast(monthly_counts)
        forecasted_data, forecasted = forecast(monthly_counts)
        fig3 = plot_timeseries(forecasted_data, forecasted)
        st.plotly_chart(fig3)
        
        # data table with search and sort
        st.subheader("100 Most Revent Event Details")
        st.dataframe(
            df.drop(columns=['event_id']).sort_values('date_of_event', ascending=False)[:100],
            use_container_width=True
        )
        
    else:
        st.info("No events found with the selected filters.")

if __name__ == "__main__":
    main()


