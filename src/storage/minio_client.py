"""MinIO client for object storage operations."""
import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Generator, Optional

from minio import Minio
from minio.error import S3Error

from config.settings import settings

logger = logging.getLogger(__name__)


class MinIOClient:
    """Client for interacting with MinIO object storage."""
    
    # Bucket names
    RAW_BUCKET = "raw-data"
    PROCESSED_BUCKET = "processed-data"
    TOKENIZED_BUCKET = "tokenized-data"
    
    def __init__(self):
        """Initialize MinIO client."""
        self.client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure
        )
        self._ensure_buckets()
    
    def _ensure_buckets(self) -> None:
        """Create required buckets if they don't exist."""
        for bucket in [self.RAW_BUCKET, self.PROCESSED_BUCKET, self.TOKENIZED_BUCKET]:
            try:
                if not self.client.bucket_exists(bucket):
                    self.client.make_bucket(bucket)
                    logger.info(f"Created bucket: {bucket}")
            except S3Error as e:
                logger.error(f"Error creating bucket {bucket}: {e}")
                raise
    
    def upload_file(self, bucket: str, object_name: str, file_path: Path) -> bool:
        """Upload a file to MinIO."""
        try:
            self.client.fput_object(bucket, object_name, str(file_path))
            logger.debug(f"Uploaded {file_path} to {bucket}/{object_name}")
            return True
        except S3Error as e:
            logger.error(f"Upload failed: {e}")
            return False
    
    def upload_json(self, bucket: str, object_name: str, data: dict) -> bool:
        """Upload JSON data to MinIO."""
        try:
            json_bytes = json.dumps(data, default=str).encode("utf-8")
            self.client.put_object(
                bucket,
                object_name,
                BytesIO(json_bytes),
                len(json_bytes),
                content_type="application/json"
            )
            return True
        except S3Error as e:
            logger.error(f"JSON upload failed: {e}")
            return False
    
    def download_file(self, bucket: str, object_name: str, file_path: Path) -> bool:
        """Download a file from MinIO."""
        try:
            self.client.fget_object(bucket, object_name, str(file_path))
            return True
        except S3Error as e:
            logger.error(f"Download failed: {e}")
            return False
    
    def list_objects(
        self, bucket: str, prefix: str = ""
    ) -> Generator[str, None, None]:
        """List objects in a bucket."""
        try:
            objects = self.client.list_objects(bucket, prefix=prefix, recursive=True)
            for obj in objects:
                yield obj.object_name
        except S3Error as e:
            logger.error(f"List objects failed: {e}")