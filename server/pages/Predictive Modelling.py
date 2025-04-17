"""
run this file with the command
    $ python -m streamlit run Home.py
to open dashboard on port 8051

NOTE: please set your own password in this file and init_recall_db.py
"""

import streamlit as st
import pandas as pd
from psycopg2.extras import RealDictCursor
import plotly.express as px
from datetime import datetime, timedelta
from db_info import get_db_connection, get_db_cursor
import pickle
import model_objects.run_algos as a
import matplotlib.pyplot as plt
import seaborn as sns

# connect to local db
def get_database_connection():
    try:
        conn = get_db_connection()
        return conn
    except Exception as e:
        st.error(f'Database connection error: {e}')
        return None
    
# function to fetch adverse event device names for dropdown
def fetch_device_names_all():
    conn = get_database_connection()
    if not conn:
        return pd.DataFrame()
    try:
        query = """
            select distinct openfda_device_name as device_name from device
        """
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            results = cur.fetchall()
        
        df = pd.DataFrame(results)
        return df
    except Exception as e:
        st.error(f'Error fetching device name list: {e}')
        return pd.DataFrame()
    finally:
        conn.close()
    
def model_data(st):   
    st.title("Predictive Modeling")
    
    st.text(f"""
        This page allows a user to select a device associated to an adverse event and predict 
        1) the probability the same device having a recall and 
        2) the type of event that is most likely to occur. 
        This is helpful for manufacturers to determine the likelihood of a particular device to have issues, allowing them to take preventative measures and prioritize safety prior to market release. 

        Both models were trained using XGBoost classifiers; the recall probability used a logistic regression implementation since it was a binary output, while the event types had multiple classes: Death, Injury, Malfunction, Unknown, and Other. The data was preprocessed based on feature type: 
        a) categorical features were one-hot encoded.
        b) numerical features were impute.
        c) text features were transformed to remove stopwords and vectorized using Term Frequency-Inverse Document Frequency (TF-IDF).

        The models were trained on 80% of the input data, which was gathered from the device event and recall related tables, and the remaining 20% is used to determine the test accuracies and confusion matrices shown below. The features included device name, approval type (PMA vs 510k), number of adverse events, and device classification. 
    """)
    
    col1, col2 = st.columns(2)
    path1 = 'model_objects/classifier/recall_probability'
    path2 = 'model_objects/classifier/event_type'
    
    labels = {0: 'Death', 1: 'Injury', 2: 'Malfunction', 3: 'Unknown/Other', 4: 'Unknown/Other'}

    # get accuracies
    accuracy1, plt1 = a.get_model_accuracy(path1)
    col1.metric('Test Accuracy for Recall Probability ', f'{accuracy1*100:.3f}%', border=True)

    accuracy2, plt2 = a.get_model_accuracy(path2)
    col2.metric('Test Accuracy for Event Type Prediction ', f'{accuracy2*100:.3f}%', border=True)
    
    # get confusion matrices
    col1, col2 = st.columns(2)
    col1.image(plt1)
    col2.image(plt2)

    # add dropdown for device name search
    df_all = fetch_device_names_all()

    if not df_all.empty:
        st.subheader('Make a Prediction')
        device_list = df_all.device_name.unique()
        # get prediction
        device_name = st.selectbox(
            'Select Device Name',
            device_list
        )
        col1_1, col2_2 = st.columns(2)

        recall_probability, _, df = a.make_prediction(device_name, path1)
        event_class, event_probabilities, _ = a.make_prediction(device_name, path2)
        event_type = labels.get(event_class, -1)
        if recall_probability != -1:
            col1_1.metric('Probability of Recall ',  f'{recall_probability*100:.3f}%')
            # plot the table of feature values
            df = df.reset_index(drop=True)[['product_code', 'device_class', 'regulation_number', 'pma_approval', '510k_approval', 'num_events']]         
            col1_1.dataframe(df.T) 

        if event_type != -1:
            col2_2.metric('Predicted Event Type ', event_type)
            # plot the bar chart of probability distribution
            prob_df = pd.DataFrame.from_dict(
                {
                'Death': [float(event_probabilities[0])],
                'Injury': [float(event_probabilities[1])],
                'Malfunction':[float(event_probabilities[2])],
                'Unknown/Other': [float(event_probabilities[3] + event_probabilities[4])]
                }
            )
            prob_df = prob_df.reset_index(drop=True)
            col2_2.bar_chart(prob_df, x_label='Probability Breakdown for Device')
    else:
        st.info("No device names found.")

def cluster_data(st):
    st.header('Clustering Manufactuer and Device Names')
    path = 'model_objects/cluster'
    df, plot = a.get_cluster_objects(path)

    col1, col2 = st.columns(2)
    # prompt user for cluster id
    cluster = int(col1.number_input('Enter Cluster Number', value=0))
    # get all items in  cluster
    get_cluster = df[df.cluster == cluster]
    col2.dataframe(get_cluster[['cluster','device_name', 'manufacturer_name']])

    
    # display cluster plot
    st.plotly_chart(plot)

# app layout and functionality
def main():
    st.set_page_config(
        page_title="OpenFDA Medical Devices - Predictions",
        page_icon="⚕️",
        layout="wide"
    )
    
    model_data(st)
    # cluster_data(st)

if __name__ == "__main__":
    main()


