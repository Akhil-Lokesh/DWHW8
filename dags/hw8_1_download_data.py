from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import requests
import pandas as pd
import os

# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 4, 17),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Initialize the DAG
dag = DAG(
    'hw8_download_process_data',
    default_args=default_args,
    description='Download and process Medium articles data for Pinecone',
    schedule_interval=None,
    catchup=False,
    tags=['homework8', 'pinecone', 'download'],
)

def download_medium_data():
    """
    Downloads Medium articles dataset from GitHub
    """
    url = "https://raw.githubusercontent.com/bkolligs/article_embeddings/main/medium_articles_sample.csv"
    response = requests.get(url)
    
    # Create the file path
    file_path = '/opt/airflow/dags/medium_articles.csv'
    
    # Save the file
    with open(file_path, 'wb') as f:
        f.write(response.content)
    
    print(f"File downloaded to: {file_path}")
    return file_path

def process_medium_data(**kwargs):
    """
    Processes the downloaded medium articles dataset and prepares it for embedding
    """
    ti = kwargs['ti']
    file_path = ti.xcom_pull(task_ids='download_medium_data')
    output_path = '/opt/airflow/dags/processed_medium_data.csv'
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Add an ID column if it doesn't exist
    if 'id' not in df.columns:
        df['id'] = df.index
    
    # Ensure title and subtitle columns exist
    if 'title' not in df.columns:
        df['title'] = 'No title'
    if 'subtitle' not in df.columns:
        df['subtitle'] = ''
    
    # Create metadata column by combining title and subtitle
    df['metadata'] = df.apply(lambda row: {'title': str(row['title']) + " " + str(row['subtitle'])}, axis=1)
    
    # Select only necessary columns
    processed_df = df[['id', 'title', 'subtitle', 'metadata']]
    
    # Save processed data
    processed_df.to_csv(output_path, index=False)
    
    print(f"Processed data saved to: {output_path}")
    print(f"Preview of processed data: {processed_df.head()}")
    
    return output_path

# Define the tasks
download_task = PythonOperator(
    task_id='download_medium_data',
    python_callable=download_medium_data,
    dag=dag,
)

process_task = PythonOperator(
    task_id='process_medium_data',
    python_callable=process_medium_data,
    provide_context=True,
    dag=dag,
)

# Set task dependencies
download_task >> process_task