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
    'hw8_run_pinecone_search',
    default_args=default_args,
    description='Run search queries against Pinecone index',
    schedule_interval=None,
    catchup=False,
    tags=['homework8', 'pinecone', 'search'],
)

def search_pinecone(query="data science", top_k=5):
    """
    Performs a semantic search against the Pinecone index
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
    
    # Get API key from Airflow Variable
    api_key = Variable.get("pinecone_api_key")
    
    # Initialize Pinecone with API key using the new API pattern
    print(f"Initializing Pinecone...")
    pc = Pinecone(api_key=api_key)
    
    # List available indexes
    try:
        available_indexes = pc.list_indexes()
        print(f"Available indexes: {available_indexes}")
        
        # Check if our index exists
        index_name = "medium-articles-index"
        if index_name not in available_indexes.names():
            raise ValueError(f"Index '{index_name}' does not exist. Please run the create_pinecone_index DAG first.")
    except Exception as e:
        print(f"Error checking existing indexes: {e}")
        raise
    
    print(f"Connecting to index: {index_name}")
    index = pc.Index(index_name)
    
    # Initialize the sentence transformer model
    print("Loading the sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate embedding for the query
    print(f"Generating embedding for query: '{query}'")
    query_embedding = model.encode(query).tolist()
    
    # Perform the search
    print(f"Searching Pinecone for: '{query}'")
    try:
        # Use the latest Pinecone API format
        search_results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Display results
        print(f"\nSearch Results for '{query}':")
        print("=" * 50)
        
        has_matches = False
        try:
            # Try with matches as attribute
            if hasattr(search_results, 'matches') and search_results.matches:
                has_matches = True
                for i, match in enumerate(search_results.matches):
                    article_title = match.metadata.get('title', 'No title available')
                    similarity_score = match.score
                    article_id = match.id
                    
                    print(f"{i+1}. Article ID: {article_id}")
                    print(f"   Title: {article_title}")
                    print(f"   Similarity Score: {similarity_score:.4f}")
                    print("-" * 50)
        except Exception as e:
            print(f"Error processing results as object with attributes: {e}")
            
        # Try with matches as dictionary
        if not has_matches:
            try:
                if 'matches' in search_results and search_results['matches']:
                    for i, match in enumerate(search_results['matches']):
                        article_title = match['metadata'].get('title', 'No title available')
                        similarity_score = match['score']
                        article_id = match['id']
                        
                        print(f"{i+1}. Article ID: {article_id}")
                        print(f"   Title: {article_title}")
                        print(f"   Similarity Score: {similarity_score:.4f}")
                        print("-" * 50)
                else:
                    print("No results found.")
            except Exception as e:
                print(f"Error processing results as dictionary: {e}")
                print(f"Raw search results: {search_results}")
    
    except Exception as e:
        print(f"Error during search: {e}")
        try:
            # Try alternate query format
            print("Trying alternative query format...")
            search_results = index.query(
                queries=None,
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            print(f"Raw search results: {search_results}")
        except Exception as e2:
            print(f"Alternative query also failed: {e2}")
            raise
    
    return f"Completed search for '{query}'"

# Define tasks for different search queries
search_task1 = PythonOperator(
    task_id='search_data_science',
    python_callable=search_pinecone,
    op_kwargs={'query': 'data science and machine learning', 'top_k': 5},
    dag=dag,
)

search_task2 = PythonOperator(
    task_id='search_business',
    python_callable=search_pinecone,
    op_kwargs={'query': 'business strategy and entrepreneurship', 'top_k': 5},
    dag=dag,
)

search_task3 = PythonOperator(
    task_id='search_technology',
    python_callable=search_pinecone,
    op_kwargs={'query': 'artificial intelligence and ethics', 'top_k': 5},
    dag=dag,
)