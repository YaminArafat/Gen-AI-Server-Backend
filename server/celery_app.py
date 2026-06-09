import os
import asyncio
import json
import time
from typing import Dict, Any
from celery import Celery
import boto3
from botocore.client import Config as BotoConfig
from server.gen_ai_service import get_orchestration_dual_chain
from services.s3_service import storage_service

redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
celery_engine = Celery(
    "config_tasks",
    broker=redis_url,
    backend=redis_url
)

GPU_CONCURRENCY_SEMAPHORE = asyncio.Semaphore(2)

# def _get_s3_data_lake_client():
#     return boto3.client(
#         's3',
#         endpoint_url=f"http://{os.getenv('MINIO_ENDPOINT', 'localhost:9000')}",
#         aws_access_key_id=os.getenv('MINIO_ACCESS_KEY', 'datalake_admin'),
#         aws_secret_access_key=os.getenv('MINIO_SECRET_KEY', 'datalake_secret_key'),
#         config=BotoConfig(signature_version='s3v4'),
#         region_name='us-east-1'
#     )

async def _gpu_bound_inference(task_id: str, user_input: str, context: str) -> Dict[str, Any]:
    async with GPU_CONCURRENCY_SEMAPHORE:
        pipeline_chain = get_orchestration_dual_chain(task_id)

        pipeline_output = await pipeline_chain.ainvoke({
                "user_input": user_input,
                "context": context
            })
        return pipeline_output

@celery_engine.task(name="tasks.process_config_generation", bind=True)
def process_config_generation(self, task_payload: Dict[str, Any]) -> Dict[str, Any]:

    task_id = self.request.id
    user_input = task_payload.get("user_input", "")
    context = task_payload.get("context", "")
    use_rag = task_payload.get("useRAG", True)
    
    start_timestamp = time.time()
    execution_error = None
    output = None
    
    try:
        output = asyncio.run(_gpu_bound_inference(task_id, user_input, context))
        return {"status": "Success", "data": output["config"]}
        
    except Exception as exc:
        execution_error = str(exc)
        return {"status": "Failed", "error": execution_error}
        
    finally:
        latency = (time.time() - start_timestamp) * 1000
        
        telemetry_record = {
            "task_id": task_id,
            "user_prompt": user_input,
            "latency_ms": latency,
            "rag_enabled": use_rag,
            "execution_status": "Success" if not execution_error else "Failed",
            "error_payload": execution_error,
            "generated_output": output,
            "timestamp": time.time()
        }
        s3_bucket_name = os.getenv("MINIO_TELEMETRY_BUCKET_NAME", "config-telemetry-landing-zone")

        storage_service._create_bucket_if_not_exists(bucket_name=s3_bucket_name)
                
        storage_service.internal_client.put_object(
            Bucket=s3_bucket_name,
            Key=f"raw_events/{task_id}_log.json",
            Body=json.dumps(telemetry_record),
            ContentType="application/json"
        )