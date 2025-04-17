"""
pip install scikit-learn nltk matplotlib cleanco levenshtein name_matching xgboost

make sure "xgboost" directory exists in path
"""
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import re
import unicodedata
from cleanco import basename
import time
from name_matching.name_matcher import NameMatcher
import pickle
import os
import psycopg2
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
import pickle
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GridSearchCV
import plotly.express as px


# # NOTE - these require internet connection
# nltk.download('stopwords')
# nltk.download('punkt_tab')

# DB params. ensure that there is an environment variable for postgres_password
DB_PARAMS = {
    "host": "localhost",
    "database": "openfda_recalls",
    "user": "postgres",
    "password": os.environ.get("postgres_password"),
    "port": 5432
}

# function to read data into dataframe
def read_table(table,query=None):
    def get_postgres_conn():
        conn = None
        try:
            conn = psycopg2.connect(**DB_PARAMS)
            return conn
        
        except psycopg2.Error as e:
            print(f"Database error: {e}")
            return pd.DataFrame()
    conn = get_postgres_conn()
    print(f'loading query into df...')
    if query is None:
        query = f'select * from {table}'
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# function to preprocess text col
def preprocess_col(df, col):
    """
    0. replace Nan with empty string
    1. lower and strip spaces
    2. remove non-ascii chars
    3. remove punctuation
    4. remove common legal business strings (like 'corp')
    """
    # remove NaN or None
    df = df.fillna('')
    
    # stop_words = set(stopwords.words('english'))
    def f(x):
        x = x.lower().strip() # step 1
        x = unicodedata.normalize('NFKD', x).encode('ASCII', 'ignore').decode() # step 2
        x = re.sub(r'[^\w\s]', '', x) # step 3
        x = basename(x) # step 4
        return x
    return df[col].apply(lambda x: f(x))

# function to preprocess data based on categorical/text/numerical cols
def preprocess_data(df, categorical, text, numerical, encoder=None, vectorizer=None, pca=None, run_pca=True):
    """
    given data df, convert
    - categorical columns to numerical
    - text to TFIDF 
    - nans in numerical cols (though at this point there shouldn't be)
    """
    ######################################
    # one hot encode categorical 
    if categorical != []:
        cat_data = df[categorical]
        if encoder is None:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').set_output(transform='pandas')
            encoder.fit(cat_data)
        cat_data_encoded = encoder.transform(cat_data)
    else: 
        cat_data_encoded = pd.DataFrame()

    ######################################
    # convert text data into numerical representation
    if text != None:
        if vectorizer is None:
            vectorizer = TfidfVectorizer()
            vectorizer.fit(df[text])
        text_data_vectorized = vectorizer.transform(df[text])
    
        # dimensionality reduction using pca and convert to df; tfidf returns sparse arrays
        if run_pca:
            if pca is None:
                # keep 500 cols
                pca = PCA(n_components=500)
                pca.fit(text_data_vectorized)
            text_data_pca = pd.DataFrame(pca.transform(text_data_vectorized))
        else:
            text_data_pca = pd.DataFrame(text_data_vectorized.toarray())
    else: 
        text_data_pca = pd.DataFrame()
    
    ######################################    
    # fill na for numerical
    if numerical != []:
        numerical_data = df[numerical]
        for c in numerical:
            numerical_data[c] = pd.to_numeric(numerical_data[c], errors='coerce').fillna(0)
    else:
        numerical_data = pd.DataFrame()

    ######################################
    # concat together
    concat_data = pd.concat([cat_data_encoded, text_data_pca, numerical_data], axis=1)
    print(concat_data.shape)

    return concat_data, encoder, vectorizer, pca

# train and save xgboost model for classification - probability of recall, event_type
def train_model():
    start = time.time()
    ############ get data ############
    print('fetching data...')
    # get device event
    df_device_event = read_table(
        'device_event',
        """
        select distinct d.manufacturer_d_name, e.event_type, device_report_product_code, d.openfda_regulation_number, d.model_number, d.openfda_device_name
        from device_event e join device d on d.event_id = e.event_id
        """
    )

    def train_recall_probability():
        # get classification
        df_class = read_table('device_classification')

        # get recall
        df_recall = read_table(
            'recall',
            '''
            select openfda_device_name, count(*) recall_count from recall group by openfda_device_name
            '''
        )

        ############ preprocess text ############
        print('initial preprocessing...')
        # preprocess names where applicable
        df_device_event['prprc_event_device_name'] = preprocess_col(df_device_event, 'openfda_device_name')
        df_class['prprc_class_device_name'] = preprocess_col(df_class, 'device_name')
        df_recall['prprc_recall_device_name'] = preprocess_col(df_recall, 'openfda_device_name')

        ############ join event data with device classification using product_code ############
        df_merged = df_device_event.merge(df_class, left_on='prprc_event_device_name', right_on='prprc_class_device_name')
        # using cols submission_type_id, determine of 510k/PMA approved
        df_merged['pma_approval'] = df_merged['submission_type_id'].apply(lambda x: 1 if x == '2' else 0)
        df_merged['510k_approval'] = df_merged['submission_type_id'].apply(lambda x: 1 if x == '1' else 0)

        ############ join with device recall using device name and get recall count ############
        df_merged = df_merged.merge(df_recall, left_on='prprc_event_device_name', right_on='prprc_recall_device_name', how='left')
        df_merged['recall_count'] = df_merged['recall_count'].fillna(0.0)
        df_merged['recall'] = df_merged['recall_count'].apply(lambda x: 1 if x > 0.0 else 0)

        ############ group to get number of event counts ############
        # group by and get counts -- removed manufacturer name because it adds unnecessary complexity 
        event_counts = df_merged.groupby(['device_name', 'product_code', 'event_type', 'device_class', 'regulation_number', 'pma_approval', '510k_approval', 'recall']).size().reset_index(name='num_events')
        # reorder columns
        event_counts = event_counts[[
            'device_name',
            'product_code',
            'event_type',
            'device_class',
            'regulation_number',
            'pma_approval',
            '510k_approval',
            'num_events',
            'recall' # output variable
        ]]

        ############ preprocess the text to numerical cols ############
        print('final preprocessing...')
        categorical = ['product_code', 'event_type', 'device_class', 'regulation_number']
        numerical = ['pma_approval', '510k_approval', 'num_events']
        # assume we only have one text col to vectorize 
        text = 'device_name' 
        data, encoder, vectorizer, pca = preprocess_data(event_counts, categorical, text, numerical)

        ########### train data ##############
        print('training...')
        # split into X and y
        X = data
        y = event_counts.iloc[:, -1]
        print(X.shape, y.shape)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=62)

        # train
        model = xgb.XGBClassifier(n_estimators=50, random_state=62, objective='binary:logistic', eval_metric='logloss') 
        model.fit(X_train, y_train)
        
        # get accuracy
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test.astype(int), predictions)
        print(f'Test Accuracy: {accuracy * 100:.2f}%')

        # generate confusion matrix
        matrix = confusion_matrix(y_test, predictions, normalize='true')
        fig = plt.figure()
        sns.heatmap(matrix, 
                    annot=True, 
                    cmap='PuRd',
                    xticklabels=['Recall=False', 'Recall=True'],
                    yticklabels=['Recall=False', 'Recall=True']
                    )
        plt.xlabel('Predicted Value')
        plt.ylabel('True Value')
        plt.title('Recall Probability - Confusion Matrix')

        print(f'\n----------- XGBoost runtime: {time.time() - start} seconds -----------')

        # save model data to pickle
        with open('model_objects/classifier/recall_probability/classifier.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open('model_objects/classifier/recall_probability/pca.pkl', 'wb') as f:
            pickle.dump(pca, f)
        with open('model_objects/classifier/recall_probability/vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
        with open('model_objects/classifier/recall_probability/encoder.pkl', 'wb') as f:
            pickle.dump(encoder, f)
        with open('model_objects/classifier/recall_probability/accuracy.pkl', 'wb') as f:
            pickle.dump(accuracy, f)
        # with open('model_objects/classifier/recall_probability/confusion_matrix.pkl', 'wb') as f:
        #     pickle.dump(fig, f)
        plt.savefig('model_objects/classifier/recall_probability/confusion_matrix.png')

    def train_event_classification():    
        ############ preprocess text ############
        print('initial preprocessing...')
        # preprocess names where applicable
        df_device_event['device_name'] = preprocess_col(df_device_event, 'openfda_device_name')
 
        ############ preprocess the text to numerical cols ############
        print('final preprocessing...')
        # assume we only have one text col to vectorize 
        categorical = []
        numerical = []
        text = 'device_name' 
        data, encoder, vectorizer, pca = preprocess_data(df_device_event, categorical, text, numerical)
        # label output
        labels = {'Death': 0, 'Injury': 1, 'Malfunction': 2, 'No answer provided': 3, 'Other': 4}
        df_device_event['y'] = df_device_event['event_type'].replace(labels)
        
        ########### train data ##############
        print('training...')
        # split into X and y
        X = data.values
        y = df_device_event['y']

        print(X.shape, y.shape)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=62)

        # # train via hyperparameter tuning
        # param_grid = {
        #     'max_depth': [2, 5, 10],
        # }
        # model = xgb.XGBClassifier() 

        # # grid search
        # grid_search = GridSearchCV(model, param_grid, scoring='accuracy')
        # grid_search.fit(X, y)
        
        # # get best model accuracy
        # model = grid_search.best_estimator_
        # accuracy = grid_search.best_score_
        # print(f'Best Accuracy in Grid: {accuracy * 100:.2f}')
        
        # train
        model = xgb.XGBClassifier()
        model.fit(X_train, y_train)

        # get accuracy
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test.astype(int), predictions)
        print(f'Test Accuracy: {accuracy * 100:.2f}%')

        # generate confusion matrix
        matrix = confusion_matrix(y_test, predictions, normalize='true')
        fig = plt.figure()
        sns.heatmap(matrix, 
                    annot=True, 
                    cmap='PuRd',
                    xticklabels=['Death', 'Injury', 'Malfunction', 'Unknown', 'Other'],
                    yticklabels=['Death', 'Injury', 'Malfunction', 'Unknown', 'Other'],
                    )
        plt.xlabel('Predicted Value')
        plt.ylabel('True Value')
        plt.title('Event Type - Confusion Matrix')

        print(f'\n----------- XGBoost runtime: {time.time() - start} seconds -----------')

        # save model data to pickle
        with open('model_objects/classifier/event_type/classifier.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open('model_objects/classifier/event_type/pca.pkl', 'wb') as f:
            pickle.dump(pca, f)
        with open('model_objects/classifier/event_type/vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
        with open('model_objects/classifier/event_type/encoder.pkl', 'wb') as f:
            pickle.dump(encoder, f)
        with open('model_objects/classifier/event_type/accuracy.pkl', 'wb') as f:
            pickle.dump(accuracy, f)        
        # with open('model_objects/classifier/event_type/confusion_matrix.pkl', 'wb') as f:
        #     pickle.dump(fig, f)
        plt.savefig('model_objects/classifier/event_type/confusion_matrix.png')

    print('\nTRAINING RECALL PROBABILITY....')
    train_recall_probability()
    
    print('\nTRAINING EVENT TYPE CLASSIFICATION....')
    train_event_classification()

# get model and encoders from pickle
def get_model_objects(path):
    with open(f'{path}/classifier.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(f'{path}/pca.pkl', 'rb') as f:
        pca = pickle.load(f)
    with open(f'{path}/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open(f'{path}/encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    with open(f'{path}/accuracy.pkl', 'rb') as f:
        accuracy = pickle.load(f)
    # with open(f'{path}/confusion_matrix.pkl', 'rb') as f:
    #     matrix = pickle.load(f)
    matrix = f'{path}/confusion_matrix.png'
    return model, encoder, vectorizer, pca, accuracy, matrix

# given a device name, get required event data from model prediction and preprocess. 
def make_prediction(device_name, path):
    # get model params
    model, encoder, vectorizer, pca, _, _ = get_model_objects(path)

    # depending on model type, get query and preprocess
    if 'recall_probability' in path:
        # get data from query
        query = """
            select distinct 
                d.openfda_device_name as device_name,
                c.product_code,
                e.event_type, 
                c.device_class,
                d.openfda_regulation_number as regulation_number,
                case when c.submission_type_id = '2' then 1 else 0 end as pma_approval,
                case when c.submission_type_id = '1' then 1 else 0 end as "510k_approval",
                count(*) as num_events
            from device d
            join device_event e
                on e.event_id = d.event_id
            join device_classification c 
                on c.device_name = d.openfda_device_name 
            where device_name = '%s'
            group by
                d.openfda_device_name,
                c.product_code,
                e.event_type, 
                c.device_class,
                d.openfda_regulation_number,
                case when c.submission_type_id = '2' then 1 else 0 end,
                case when c.submission_type_id = '1' then 1 else 0 end,
                d.openfda_device_name
        """ % device_name
        df = read_table('device_event', query)
        # if no data found, return -1
        if df.empty:
            return -1, None
        print(df)

        # preprocess input
        categorical = ['product_code', 'event_type', 'device_class', 'regulation_number']
        numerical = ['pma_approval', '510k_approval', 'num_events']
        text = 'device_name' 
        X, encoder, vectorizer, pca = preprocess_data(df, categorical, text, numerical, encoder, vectorizer, pca)

        # make prediction 
        prediction =  model.predict_proba(X)
        # if we have more than one row for device_name keep first prediction
        return prediction[0][1], prediction[0], df
    
    elif 'event_type' in path:
        # get data from query
        query = """
            select distinct d.manufacturer_d_name, e.event_type, device_report_product_code, d.openfda_regulation_number, d.model_number, d.openfda_device_name as device_name
            from device_event e join device d on d.event_id = e.event_id 
            where openfda_device_name = '%s'

        """ % device_name
        df = read_table('device_event', query)
        # if no data found, return -1
        if df.empty:
            return -1, None
        print(df)

        # preprocess input
        text = 'device_name' 
        X, encoder, vectorizer, pca = preprocess_data(df, [], text, [], encoder, vectorizer, pca)

        # make prediction 
        prediction =  model.predict(X)
        # if we have more than one row for device_name keep first prediction
        return prediction[0], model.predict_proba(X)[0], df

# get model accuracy and generate confusion  matrix to display
def get_model_accuracy(path):
    # get model accuracy 
    _, _, _, _, accuracy, matrix_plt = get_model_objects(path)
    return accuracy, matrix_plt

# train and save cluster
def create_cluster():
    start = time.time()
    ############ get data ############
    print('fetching data...')
    # get event manufacturer and device name from event and recalls
    df = read_table(
        'device_event',
        """
        select distinct * from (
            select distinct manufacturer_d_name as manufacturer_name, openfda_device_name as device_name 
                from device
            UNION
            select distinct recalling_firm, openfda_device_name  
                from recall   
        ) x
        """
    )

    ############ preprocess text ############
    print('initial preprocessing...')
    # preprocess names where applicable
    df['prprc_device_name'] = preprocess_col(df, 'device_name')
    df['prprc_manufacturer_name'] = preprocess_col(df, 'manufacturer_name')

    ############ preprocess the text to numerical cols ############
    print('final preprocessing...')
    # concat the two cols before vectorization
    text = 'concat_name' 
    df[text] = df['prprc_device_name'] + df['prprc_manufacturer_name']
    data, _, vectorizer, _ = preprocess_data(df, [], text, [], run_pca=False)

    ########### cluster data ##############
    print('clustering...')  
    # calculate cosing similarity of all values in df
    similarity = cosine_similarity(data)
    # cluster using dbscan
    cluster = DBSCAN(eps=0.1, min_samples=1, metric='cosine').fit(similarity)
    labels = cluster.labels_
    unique, counts = np.unique(labels, return_counts = True)
    print('num clusters:', len(unique))

    # append cluster id to dataframe 
    df['cluster'] = labels
   
    # generate cluster plot by running pca
    print('running PCA on cluster data...')
    X_embedded = PCA(n_components=2).fit_transform(data)
    df["x_component"]=X_embedded[:,0]
    df["y_component"]=X_embedded[:,1]
    
    print('generating cluster plot...')
    fig = px.scatter(df, x="x_component", y="y_component", color = "cluster", size_max=60)    

    print(f'----------- clustering runtime: {time.time() - start} seconds -----------')
    
    # save cluster and encoders to pickle
    with open('model_objects/cluster/cluster.pkl', 'wb') as f:
        pickle.dump(cluster, f)
    with open('model_objects/cluster/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('model_objects/cluster/cluster_df.pkl', 'wb') as f:
        pickle.dump(df, f) 
    with open(f'model_objects/cluster/cluster_plot.pkl', 'wb') as f:
        pickle.dump(fig, f)

# get cluster df, model, and encoders from pickle
def get_cluster_objects(path):
    with open(f'{path}/cluster_df.pkl', 'rb') as f:
        df = pd.read_pickle(f)
    with open(f'{path}/cluster_plot.pkl', 'rb') as f:
        plot = pickle.load(f)
    return df, plot

# run functions
def run(prompt=True):
    if prompt:
        run_xgboost = input('\nCreate XGBoost Model (Y/n)?  ')
        if run_xgboost.lower().strip() == 'y':
            train_model()
            print('XGBoost model saved!')
        
        run_cluster = input('\nCreate Cluster (Y/n)?  ')
        if run_cluster.lower().strip() == 'y':
            create_cluster()
            print('Cluster saved!')

        print('\nGoodbye!')
    else:
        train_model()
        # create_cluster()

if __name__ == '__main__':
    run(prompt=False) # set to True if user wants more control over which algo to run