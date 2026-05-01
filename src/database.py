import sqlite3
import pandas as pd

DB_PATH = "data/air_quality.db"

def save_data(df):
    
    
    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql('aqi_records', conn, if_exists='replace', index=False)
    print("Data Saved.")

def load_data():

    with sqlite3.connect(DB_PATH) as conn:
        print('Hello world')
        return pd.read_sql('SELECT * FROM aqi_records', conn)