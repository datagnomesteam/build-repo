"""
this file reads data from the db into Python dataframes for easier analysis

prereq:
pip install distance python-Levenshtein
"""

import pandas as pd
import distance
import Levenshtein as L
from db_info import get_db_connection, get_db_cursor


def get_postgres_conn():
    conn = None
    try:
        conn = get_db_connection()
        return conn
    
    except Exception as e:
        print(f"Database error: {e}")
        return pd.DataFrame()

def read_table(table, conn):
    print(f'loading table: {table}....')
    df = pd.read_sql_query(f'select * from {table}', conn)
    print(df.head())
    return df

def get_dfs():
    import time
    start = time.time()
    conn = get_postgres_conn()

    df_recall = read_table('recall', conn)
    df_device_event = read_table('device_event', conn)
    df_device = read_table('device', conn)

    conn.close()
    print(f'\n----- elapsed time: {time.time() - start} seconds -----')

    return df_recall, df_device_event, df_device

if __name__ == '__main__':
    df_recall, df_device_event, df_device = get_dfs()

    # get similarity score between: device.manufacturer_d_name and recall.recalling_firm
    recalling_firms = [x.lower() for x in df_recall['recalling_firm'].unique() if x is not None]
    manufacturer_d_name = [x.lower() for x in df_device['manufacturer_d_name'].unique() if x is not None]

    # maybe ignore common words like 'medical', 'inc', 'corporation' and strip all non-alpha chars

    # jd = lambda x, y: 1 - distance.jaccard(x, y)
    ls = lambda x, y: L.ratio(x, y)

    best_matches = {}
    for a in recalling_firms:
        for b in manufacturer_d_name:
            score = ls(a, b)
            if score >= 0.95:
                key = (a, b)
                if best_matches.get(key) is not None:
                    val = best_matches[key]
                    if val < score:
                        best_matches[key] = score
                else:
                    best_matches[key] = score

    print(f'found {len(best_matches)} matches')

    # get joined data
    # df_merged_manufacturer = pd.merge(
    #     df_recall,
    #     df_device,
    #     left_on = df_recall['recalling_firm'].str.lower(),
    #     right_on = df_device['manufacturer_d_name'].str.lower()
    # )

    conn = get_postgres_conn()
    df_manufacturer_counts = pd.read_sql_query(
        """ 
        select event_mname, event_c as event_counts, recall_c as recall_counts from 
            (
                select lower(manufacturer_d_name) event_mname, count(*) event_c
                from device
                group by manufacturer_d_name
            ) d
            join 	(
                select lower(recalling_firm) recall_mname, count(*) recall_c
                from recall
                group by recalling_firm
            ) r
            
        on d.event_mname = r.recall_mname
        """
        , conn
    )
    conn.close()

    print(df_manufacturer_counts)

