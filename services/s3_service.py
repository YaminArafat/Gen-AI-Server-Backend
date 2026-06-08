import os
import io
import boto3
from botocore.config import Config as BotoConfig
from typing import BinaryIO

class S3Service:
    def __init__(self):
        self.bucket_name = os.getenv("MINIO_BUCKET_NAME", "ai-assets")
        access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
        
        self.internal_client = boto3.client(
            "s3",
            endpoint_url=os.getenv("MINIO_INTERNAL_URL", "http://minio:9000"),
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=BotoConfig(signature_version="s3v4"),
            region_name="us-east-1"
        )
        
        self.public_client = boto3.client(
            "s3",
            endpoint_url=os.getenv("MINIO_PUBLIC_URL", "http://localhost:9000"),
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=BotoConfig(signature_version="s3v4"),
            region_name="us-east-1"
        )
        
        self._ensure_bucket_exists()

    def _create_bucket_if_not_exists(self, bucket_name: str):
        try:
            self.internal_client.head_bucket(Bucket=bucket_name)
        except self.internal_client.exceptions.ClientError:
            print(f"Bucket '{bucket_name}' not found. Creating it now...")
            self.internal_client.create_bucket(Bucket=bucket_name)

    def _ensure_bucket_exists(self):
        try:
            self.internal_client.head_bucket(Bucket=self.bucket_name)
        except self.internal_client.exceptions.ClientError:
            print(f"Bucket '{self.bucket_name}' not found. Creating it now...")
            self.internal_client.create_bucket(Bucket=self.bucket_name)

    def upload_file_stream(self, file_data: io.BytesIO, object_name: str, content_type: str) -> str:
        file_data.seek(0)  # Ensure stream pointer is at the beginning
        self.internal_client.put_object(
            Bucket=self.bucket_name,
            Key=object_name,
            Body=file_data,
            ContentType=content_type
        )
        return object_name

    def get_presigned_url(self, object_name: str, expires_in_seconds: int = 86400) -> str:
        return self.public_client.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": self.bucket_name, "Key": object_name},
            ExpiresIn=expires_in_seconds
        )

storage_service = S3Service()