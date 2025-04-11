# Summary

After loading data into PostgreSQL, we need to integrate and pre-aggregate the tables to create interesting views for the user.

### Install Dependencies

```
pip install streamlit pandas sqlalchemy psycopg2 pyspark
```

Note: Running PySpark will require having Java properly installed and configured in the relevant environment.

### Integrate Data

The tables are integrated according to a set of common canonical fields using PySpark.

```
python integrate.py
```

### Create Views

We use the psycopg2 PostgreSQL database adaptor to built permanent views of the table. One for unique manufacturers, one for unique devices, and one for manufacturer addresses.

```
python create_views.py
```

### Create Dashboard

Finally, we use a streamlit dashboard to interact with the views. We read the data from PostgreSQL into Pandas dataframes, and configure them to cross filter, such that a search on dataframe will filter another. 

```
streamlit run app.py
```