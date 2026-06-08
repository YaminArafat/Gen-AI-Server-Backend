from datetime import datetime, timedelta
import os
import sqlite3
from langchain.schema import Document
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from pathlib import Path

import psycopg2

from server.rag_vector_store import add_examples_to_rag

DB_PATH = Path("/app/datasets/telemetry.db")
ANALYTICS_OUTPUT_PATH = Path("/app/datasets/daily_performance_metrics.csv")

default_args = {
    'owner': 'yamin_arafat',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email_on_failure': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

def extract_and_transform_telemetry():
    if not DB_PATH.exists():
        print("No telemetry records database detected. Skipping analytics.")
        return

    with sqlite3.connect(str(DB_PATH)) as conn:
        df = pd.read_sql_query("SELECT * FROM model_telemetry", conn)

    if df.empty:
        print("Telemetry tables are completely empty.")
        return

    total_requests = len(df)
    failed_requests = len(df[df['status'] == 'Failed'])
    average_latency = df['latency_ms'].mean()
    rag_utilization = df['rag_enabled'].sum() / total_requests if total_requests > 0 else 0

    summary_data = {
        "execution_date": [datetime.now().strftime("%Y-%m-%d")],
        "total_volume": [total_requests],
        "failed_volume": [failed_requests],
        "success_rate": [(total_requests - failed_requests) / total_requests * 100],
        "mean_latency_ms": [average_latency],
        "rag_activation_percentage": [rag_utilization * 100]
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save or append calculations to a tracking file dashboards
    if ANALYTICS_OUTPUT_PATH.exists():
        summary_df.to_csv(ANALYTICS_OUTPUT_PATH, mode='a', header=False, index=False)
    else:
        summary_df.to_csv(ANALYTICS_OUTPUT_PATH, index=False)
        
    print(f"Successfully analyzed {total_requests} system execution requests.")

def extract_high_performing_outputs_to_rag():

    db_url = os.getenv("DATABASE_URL", "postgresql://admin:secret_password@postgres:5432/config_analytics")
    
    try:
        connection = psycopg2.connect(db_url)
        cursor = connection.cursor()
        
        # Pull high-performing outputs to update rag vector store context models
        cursor.execute("""
            SELECT user_prompt, generated_output 
            FROM processed_telemetry_records 
            WHERE execution_status = 'Success' AND latency_ms < 2000 
            AND generated_output IS NOT NULL
            ORDER BY event_time DESC
            LIMIT 10;
        """)
        if optimal_records:
            optimal_records = cursor.fetchall()
        
        # Iterate over records and append to your Chroma Vector Database store
        add_examples_to_rag(optimal_records)
        
        print(f"Vector store re-indexed with {len(optimal_records)} historical target assets.")
        cursor.close()
        connection.close()
    except Exception as err:
        print(f"Skipping re-indexing pass due to database connection error: {err}")

with DAG(
    'config_gen_ai_telemetry_pipeline',
    default_args=default_args,
    description='Automated operational health data logging and system metric aggregation',
    schedule_interval='@daily',
    catchup=False,
) as dag:

    execute_pyspark_analytics = BashOperator(
        task_id='execute_pyspark_distributed_analytics',
        bash_command='python3 /app/scripts/spark_telemetry_etl.py',
    )

    reindex_vector_store_memory = PythonOperator(
        task_id='dynamic_rag_feedback_reindex',
        python_callable=extract_high_performing_outputs_to_rag,
    )

    # compute_telemetry_metrics = PythonOperator(
    #     task_id='extract_and_transform_telemetry',
    #     python_callable=extract_and_transform_telemetry,
    # )

    execute_pyspark_analytics >> reindex_vector_store_memory
    # compute_telemetry_metrics