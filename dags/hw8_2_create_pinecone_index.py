from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from datetime import datetime, timedelta

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
    'hw8_create_pinecone_index',
    default_args=default_args,
    description='Create Pinecone index for vector search',
    schedule_interval=None,
    catchup=False,
    tags=['homework8', 'pinecone', 'index'],
)

def create_pinecone_index():
    """
    Creates a Pinecone index for vector search using the latest Pinecone API
    """
    try:
        # Import using the new API pattern
        from pinecone import Pinecone, ServerlessSpec
        print("Successfully imported pinecone package with new API")
    except Exception as e:
        print(f"Error importing pinecone with new API: {e}")
        raise Exception(f"Failed to import Pinecone packages: {e}")
    
    # Get API key from Airflow Variable
    try:
        api_key = Variable.get("pinecone_api_key")
        print(f"Successfully retrieved API key from Airflow Variable")
        
        # Print first/last few characters to verify without exposing full key
        masked_key = f"{api_key[:5]}...{api_key[-5:]}" if len(api_key) > 10 else "***key too short***"
        print(f"API key format check: {masked_key}")
    except Exception as e:
        print(f"Error retrieving API key: {e}")
        raise Exception("Failed to retrieve Pinecone API key from Airflow Variable 'pinecone_api_key'")
    
    print(f"Initializing Pinecone with API key")
    
    # Initialize Pinecone with API key using the new API pattern
    pc = Pinecone(api_key=api_key)
    
    # Define index name
    index_name = "medium-articles-index"
    
    # Check if index already exists
    try:
        existing_indexes = pc.list_indexes()
        print(f"Existing indexes: {existing_indexes}")
        
        # Check if our index exists
        if hasattr(existing_indexes, 'names') and callable(existing_indexes.names):
            names = existing_indexes.names()
            if index_name in names:
                print(f"Index '{index_name}' already exists. Skipping creation.")
                return index_name
    except Exception as e:
        print(f"Error checking existing indexes: {e}")
        print("This could indicate an invalid API key or network issue.")
        # Continue to creation attempt
    
    # Create a new index
    print(f"Creating new index: {index_name}")
    try:
        # Create index with ServerlessSpec using AWS (supported by free tier)
        pc.create_index(
            name=index_name,
            dimension=384,  # all-MiniLM-L6-v2 embedding model uses 384 dimensions
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"Successfully created index using ServerlessSpec")
    except Exception as e:
        print(f"Error with ServerlessSpec creation: {e}")
        print("Trying alternative creation method...")
        try:
            # Always include a spec - it's required in the latest API
            pc.create_index(
                name=index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print(f"Successfully created index with alternative parameters")
        except Exception as e3:
            print(f"All attempts to create index failed: {e3}")
            print("Please check your Pinecone API key. The error indicates it may be invalid.")
            raise
    
    print(f"Successfully created Pinecone index: {index_name}")
    return index_name

# Define the task
create_index_task = PythonOperator(
    task_id='create_pinecone_index',
    python_callable=create_pinecone_index,
    dag=dag,
)