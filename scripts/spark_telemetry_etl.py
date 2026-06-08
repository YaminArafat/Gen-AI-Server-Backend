import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, mean, count, when, window, current_timestamp
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, BooleanType, TimestampType

def run_distributed_telemetry_etl():
    
    # Initializes a localized Apache Spark session, pulls raw log files from MinIO,
    # calculates operational health statistics, and commits clean data to PostgreSQL.
    # Initialize unified parallel execution engine context
    spark = SparkSession.builder \
        .appName("ConfigTelemetryETLEngine") \
        .master("local[*]") \
        .config("spark.hadoop.fs.s3a.endpoint", f"http://{os.getenv('MINIO_ENDPOINT', 'localhost:9000')}") \
        .config("spark.hadoop.fs.s3a.access.key", os.getenv("MINIO_ACCESS_KEY", "datalake_admin")) \
        .config("spark.hadoop.fs.s3a.secret.key", os.getenv("MINIO_SECRET_KEY", "datalake_secret_key")) \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .getOrCreate()

    # Define explicit JSON target deserialization structure
    telemetry_schema = StructType([
        StructField("task_id", StringType(), True),
        StructField("user_prompt", StringType(), True),
        StructField("latency_ms", DoubleType(), True),
        StructField("rag_enabled", BooleanType(), True),
        StructField("execution_status", StringType(), True),
        StructField("generated_output", StringType(), True),
        StructField("error_payload", StringType(), True),
        StructField("timestamp", DoubleType(), True)
    ])

    print("Loading batch stream partitions from MinIO Object Lake repository...")
    
    # Extract: Ingest multi-line JSON logs simultaneously in parallel memory workers
    raw_df = spark.read.json("s3a://config-telemetry-landing-zone/raw_events/*.json", schema=telemetry_schema)
    
    if raw_df.count() == 0:
        print("Empty telemetry log encountered. Stopping Spark context.")
        spark.stop()
        return

    # Transform: Enforce temporal data types and clean null strings
    cleaned_df = raw_df.withColumn("event_time", col("timestamp").cast(TimestampType())) \
                       .na.fill({"error_payload": "None", "user_prompt": "Empty Prompt"})

    # DATA STREAM A: Write every individual record out to our Granular Fact Table
    cleaned_df.select(
        "task_id", "user_prompt", "latency_ms", "rag_enabled", 
        "execution_status", "error_payload", "generated_output", "event_time"
    ).write \
     .format("jdbc") \
     .option("url", postgres_url) \
     .option("dbtable", "processed_telemetry_records") \
     .option("driver", "org.postgresql.Driver") \
     .mode("append") \
     .save()

    # Aggregate: Calculate system health metrics over the processing block
    operational_metrics_df = cleaned_df.groupBy() \
        .agg(
            count("task_id").alias("total_request_volume"),
            mean("latency_ms").alias("average_latency_computation_ms"),
            count(when(col("execution_status") == "Failed", True)).alias("total_failed_requests"),
            count(when(col("rag_enabled") == True, True)).alias("rag_active_count")
        ) \
        .withColumn("processing_window_timestamp", current_timestamp())

    print("Committing structural metrics to target PostgreSQL Database instance...")
    
    # DATA STREAM B: Perform operational aggregations for dashboards
    postgres_url = f"jdbc:{os.getenv('DATABASE_URL', 'postgresql://admin:secret_password@postgres:5432/config_analytics')}"
    
    operational_metrics_df.write \
        .format("jdbc") \
        .option("url", postgres_url) \
        .option("dbtable", "aggregated_system_health_metrics") \
        .option("driver", "org.postgresql.Driver") \
        .mode("append") \
        .save()

    print("PySpark Distributed Execution Task completed successfully.")
    spark.stop()

if __name__ == "__main__":
    run_distributed_telemetry_etl()