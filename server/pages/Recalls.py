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
from db_info import get_db_connection, get_db_cursor

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

        # get recall counts by device_class
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT device_class, COUNT(*) as count 
                FROM recall 
                GROUP BY device_class 
                ORDER BY device_class
            """)
            stats['class_counts'] = cur.fetchall()
               
        # top recalling_firm
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT recalling_firm, COUNT(*) as count 
                FROM recall 
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
                FROM recall 
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

# app layout and functionality
def main():
    st.set_page_config(
        page_title="OpenFDA Medical Devices - Recalls",
        page_icon="⚕️",
        layout="wide"
    )
    
    st.title("Recalls Dashboard")
    
    # sidebar filters
    st.sidebar.header("Filters")
    
    # date range filter
    st.sidebar.subheader("Date Range")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        date_from = st.date_input("From", datetime.now() - timedelta(days=365))
    with col2:
        date_to = st.date_input("To", datetime.now())
    
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
        df = fetch_recalls()
    
    # display data and visualizations
    if not df.empty:
        # key metrics at the top
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        
        with metrics_col1:
            st.metric("Total Recalls", df.shape[0])
                
        # with metrics_col2:
        #     class_counts = df['root_cause_description'].value_counts()
        #     class_i_count = class_counts.get('Class I', 0)
        #     st.metric("Class I Recalls (Highest Risk)", class_i_count)

            
        with metrics_col3:
            recent_recalls = df[df['event_date_posted'] >= (datetime.date(datetime.now()) - timedelta(days=30))].shape[0]
            st.metric("Recalls in Last 30 Days", recent_recalls)
        
        # visualizations
        st.subheader("Recall Statistics")
        
        col1, col2 = st.columns(2, gap='medium')
    
        with col1:
            # pie chart for recall device_class
            fig1 = px.pie(
                df, 
                names='device_class', 
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
        df['event_date_posted'] = pd.to_datetime(df['event_date_posted'], errors='coerce')
        monthly_counts = df.groupby(pd.Grouper(key='event_date_posted', freq='M'), dropna=True).size().reset_index(name='count')
        
        fig3 = px.line(
            monthly_counts, 
            x='event_date_posted', 
            y='count', 
            title='Recalls Over Time',
            labels={'event_date_posted': 'Date', 'count': 'Number of Recalls'}
        )
        st.plotly_chart(fig3)
        
        # data table with search and sort
        st.subheader("100 Most Recent Event Details")
        st.dataframe(
            df.drop(columns=['id']).sort_values('event_date_posted', ascending=False)[:100],
            use_container_width=True
        )
        
    else:
        st.info("No recalls found with the selected filters.")

if __name__ == "__main__":
    main()


