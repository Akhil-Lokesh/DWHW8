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
    'hw8_create_pinecone_index_simple',
    default_args=default_args,
    description='Create Pinecone index for vector search (simple version)',
    schedule_interval=None,
    catchup=False,
    tags=['homework8', 'pinecone', 'index'],
)

def create_pinecone_index():
    """
    Creates a Pinecone index for vector search using the latest Pinecone API
    """
    try:
        # Import the Pinecone classes according to the latest API
        from pinecone import Pinecone, ServerlessSpec
        
        # Get API key from Airflow Variable with better error handling
        try:
            api_key = Variable.get("pinecone_api_key")
            print(f"Successfully retrieved API key from Airflow Variable")
            
            # Print first/last few characters to verify without exposing full key
            masked_key = f"{api_key[:5]}...{api_key[-5:]}" if len(api_key) > 10 else "***key too short***"
            print(f"API key format check: {masked_key}")
        except Exception as e:
            print(f"Error retrieving API key: {e}")
            raise Exception("Failed to retrieve Pinecone API key from Airflow Variable 'pinecone_api_key'")
        
        # Initialize Pinecone with API key using the new client pattern
        pc = Pinecone(api_key=api_key)
        
        # Define index name
        index_name = "medium-articles-index"
        
        # Check if index already exists
        try:
            existing_indexes = pc.list_indexes()
            print(f"Existing indexes: {existing_indexes}")
            
            if index_name in existing_indexes.names():
                print(f"Index '{index_name}' already exists. Skipping creation.")
                return index_name
        except Exception as e:
            print(f"Error checking indexes: {e}")
            # Continue to creation attempt
        
        # Create a new index using ServerlessSpec for free tier
        try:
            pc.create_index(
                name=index_name,
                dimension=384,  # all-MiniLM-L6-v2 embedding model uses 384 dimensions
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",  # Using AWS for free tier
                    region="us-east-1"
                )
            )
            print(f"Successfully created index with ServerlessSpec")
        except Exception as e:
            print(f"Error with ServerlessSpec creation: {e}")
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
                print(f"Successfully created index with simplified parameters")
            except Exception as e2:
                print(f"Error with simplified creation: {e2}")
                raise
        
        print(f"Successfully created Pinecone index: {index_name}")
        return index_name
    except Exception as e:
        print(f"Error in create_pinecone_index: {e}")
        import traceback
        traceback.print_exc()
        raise

# Define the task
create_index_task = PythonOperator(
    task_id='create_pinecone_index',
    python_callable=create_pinecone_index,
    dag=dag,
)