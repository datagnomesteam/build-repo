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
            select distinct openfda_device_name as device_name 
            from device d
            join device_event e
                on e.event_id = d.event_id
            where event_type is not null
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

# convert data types to Arrow-compatible types
def fix_dataframe(df):
    df = df.copy()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)
    for col in df.select_dtypes(include=['datetime64']).columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    for col in df.select_dtypes(include=['float', 'int', 'bool']).columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # df = df.fillna({
    #     col: '' if dtype.kind in 'OSU' else 0  # String columns get empty string, numeric get 0
    #     for col, dtype in df.dtypes.items()
    # })
    
    return df

# show model information
def model_data(st):   
    st.title("Predictive Modeling")
    col1, col2 = st.columns(2)
    col1.text('')
    col1.text('')

    col1.write(f"""
        This page allows a user to select a device associated with an adverse event and predict 
        1) The probability the same device having a recall and 
        2) The type of event that is most likely to occur. 
               
        This is helpful for manufacturers to determine the likelihood of a particular device to have issues, allowing them to take preventative measures and prioritize safety prior to market release. 

        Both models were trained using XGBoost implementations of Logistic Regression, since it is capable of handling complex features, categorical variable encoding, and null values, as opposed to traditional Logistic Regrssion. Recall probability is a binary output; however, event types had multiple classes - Death, Injury, Malfunction, Not Provided, and Other - and it's possible for a device to be associated with multiple types. For this reason, a model was trained for each event type and wrapped using Scikit-Learn's MultiOutputClassifier, and the result with the highest probability is displayed. 
        
        The data was preprocessed based on feature type: 
        a) Categorical features were one-hot encoded.
        b) Numerical features were imputed.
        c) Text features were transformed to remove stopwords and vectorized using Term Frequency-Inverse Document Frequency (TF-IDF).
               
        Principal Component Analysis (PCA) was used for dimensionality reduction. The models were trained on 80% of the input data, which was gathered from the device event and recall related tables, and the remaining 20% is used to determine the test accuracies and confusion matrices shown below. Note that the event type classifier's matrix represents the aggregate of all models. The features included device name, approval type (PMA vs 510k), number of adverse events, and device classification. Select the tabs on the right to toggle between the two models.
               
        Select a device name below to generate a prediction for recall probability and most likely event type.
            
    """)
    
    tab1, tab2 = col2.tabs(['Recall Probability', 'Predicted Event Type'])
    path1 = 'model_objects/classifier/recall_probability'
    path2 = 'model_objects/classifier/event_type'
    
    labels = ['Death', 'Injury', 'Malfunction', 'Not Provided', 'Other']

    # get accuracies and show
    accuracy1, plt1 = a.get_model_accuracy(path1)
    accuracy2, plt2 = a.get_model_accuracy(path2)

    tab1.metric('Test Accuracy for Recall Probability', f'{accuracy1*100:.3f}%')

    a1, a2, a3, a4, a5 = tab2.tabs(labels)
    a1.metric('Test Accuracy', f'{accuracy2[0] * 100:.3f}%')
    a2.metric('Test Accuracy', f'{accuracy2[1] * 100:.3f}%')
    a3.metric('Test Accuracy', f'{accuracy2[2] * 100:.3f}%')
    a4.metric('Test Accuracy', f'{accuracy2[3] * 100:.3f}%')
    a5.metric('Test Accuracy', f'{accuracy2[4] * 100:.3f}%')

    # get confusion matrices
    tab1.image(plt1)
    tab2.image(plt2)

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
        event_type = labels[event_class]

        if recall_probability != -1:
            df = fix_dataframe(df)
            col1_1.metric('Probability of Recall ',  f'{recall_probability*100:.3f}%')
            # plot the table of feature values
            df = df.groupby(['product_code', 'device_class', 'regulation_number', 'pma_approval', '510k_approval'])['num_events'].sum().reset_index()
            df.columns = ['product_code', 'device_class', 'regulation_number', 'pma_approval', '510k_approval', 'total_events']
            print(df)
            col1_1.dataframe(df.T) 

            # # plot probability as a pie chart
            # fig, ax = plt.subplots()
            # ax.pie([recall_probability, 1-recall_probability], colors=['lightcoral', 'lightgreen'], labels=['P(Recall=1)', 'P(Recall=0)'], autopct='%1.3f%%')
            # fig.patch.set_alpha(0)
            # col1_1.pyplot(fig)

        else: col1_1.text('No data found.')

        if event_type != -1:
            col2_2.metric('Predicted Event Type ', event_type)
            # plot the bar chart of probability distribution
            
            prob_df = pd.DataFrame.from_dict(
                {
                'Death': [float(event_probabilities[0][0][1])],
                'Injury': [float(event_probabilities[1][0][1])],
                'Malfunction':[float(event_probabilities[2][0][1])],
                'Not Provided': [float(event_probabilities[3][0][1])],
                'Other': [float(event_probabilities[4][0][1])]
                }
            )
            print(prob_df)
            prob_df = prob_df.reset_index(drop=True)
            col2_2.bar_chart(prob_df.T, x_label='Probability Breakdown for Device')
            
        else: col2_2.text('No data found')

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


