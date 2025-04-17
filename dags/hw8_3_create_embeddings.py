from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from datetime import datetime, timedelta
import pandas as pd
import json
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
    'hw8_create_embeddings_ingest',
    default_args=default_args,
    description='Create embeddings and ingest into Pinecone',
    schedule_interval=None,
    catchup=False,
    tags=['homework8', 'pinecone', 'embeddings'],
)

def generate_and_ingest_embeddings():
    """
    Generates sentence embeddings for Medium articles and ingests them into Pinecone
    """
    # Import packages inside the function to avoid import errors
    try:
        # Import using the new API pattern
        from pinecone import Pinecone
        print("Successfully imported pinecone package with new API")
    except Exception as e:
        print(f"Error importing pinecone with new API: {e}")
        raise Exception(f"Failed to import Pinecone packages: {e}")
            
    from sentence_transformers import SentenceTransformer
    
    # Check if input file exists
    file_path = '/opt/airflow/dags/processed_medium_data.csv'
    if not os.path.exists(file_path):
        alt_path = '/opt/airflow/dags/processed_medium.csv'
        if os.path.exists(alt_path):
            file_path = alt_path
            print(f"Using alternative file path: {file_path}")
        else:
            raise FileNotFoundError(f"Neither {file_path} nor {alt_path} exist. Make sure to run the download_process_data DAG first.")
    
    # Load the processed data
    df = pd.read_csv(file_path)
    
    print(f"Loaded processed data with {len(df)} records")
    print(f"Columns in dataframe: {df.columns.tolist()}")
    print(f"Sample data: {df.head(1).to_dict('records')}")
    
    # Initialize the sentence transformer model
    print("Loading the sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Get API key from Airflow Variable
    api_key = Variable.get("pinecone_api_key")
    
    # Initialize Pinecone with API key using the new API pattern
    print("Initializing Pinecone...")
    pc = Pinecone(api_key=api_key)
    
    # Connect to the index
    index_name = "medium-articles-index"
    print(f"Connecting to index: {index_name}")
    index = pc.Index(index_name)
    
    # Process records in batches to avoid memory issues
    batch_size = 16  # Smaller batch size to avoid memory issues
    total_records = len(df)
    
    print(f"Processing {total_records} records in batches of {batch_size}")
    
    for i in range(0, total_records, batch_size):
        end_idx = min(i + batch_size, total_records)
        batch = df.iloc[i:end_idx]
        
        print(f"Processing batch {i//batch_size + 1}/{(total_records + batch_size - 1)//batch_size}")
        
        # Parse metadata if stored as string
        try:
            if isinstance(batch['metadata'].iloc[0], str):
                batch['metadata'] = batch['metadata'].apply(json.loads)
                print("Converted metadata from string to dict")
        except Exception as e:
            print(f"Error parsing metadata: {e}")
            print("Attempting to create new metadata field")
            batch['metadata'] = batch.apply(
                lambda row: {'title': str(row.get('title', '')) + " " + str(row.get('subtitle', ''))}, 
                axis=1
            )
        
        # Extract titles for embedding
        try:
            titles = batch.apply(lambda row: row['metadata']['title'], axis=1).tolist()
        except Exception as e:
            print(f"Error extracting titles: {e}")
            titles = batch['title'].astype(str).tolist()
        
        # Generate embeddings
        print(f"Generating embeddings for batch {i//batch_size + 1}")
        embeddings = model.encode(titles)
        
        # Prepare records for upsert
        records = []
        for j, (_, row) in enumerate(batch.iterrows()):
            record = {
                'id': str(row['id']),
                'values': embeddings[j].tolist(),
                'metadata': row['metadata']
            }
            records.append(record)
        
        # Upsert to Pinecone
        print(f"Upserting batch {i//batch_size + 1} to Pinecone")
        try:
            # Use the new Pinecone API format
            index.upsert(
                vectors=records,
                namespace=""  # Default namespace
            )
            print(f"Upserted batch {i//batch_size + 1} with new API format")
        except Exception as e:
            print(f"Error with upsert: {e}")
            try:
                # Try alternative format if needed
                index.upsert(
                    vectors=records
                )
                print(f"Upserted batch {i//batch_size + 1} with simplified format")
            except Exception as e2:
                print(f"All upsert attempts failed: {e2}")
                raise
    
    # Get stats to verify ingestion
    try:
        stats = index.describe_index_stats()
        print(f"Pinecone index stats after ingestion: {stats}")
    except Exception as e:
        print(f"Could not get index stats: {e}")
    
    return f"Successfully ingested {total_records} records into Pinecone"

# Define the task
embedding_task = PythonOperator(
    task_id='generate_and_ingest_embeddings',
    python_callable=generate_and_ingest_embeddings,
    dag=dag,
)