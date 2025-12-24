"""
MinIO/S3 storage client for the arXiv LLM pipeline.

Provides async operations for object storage.
"""

import asyncio
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO, Optional

from minio import Minio
from minio.error import S3Error

from config.settings import settings
from src.logging_config import get_logger

logger = get_logger(__name__)


class MinioStorage:
    """
    MinIO/S3 storage client with async support.
    
    Features:
    - Bucket management
    - Streaming uploads/downloads
    - Object listing with pagination
    - Presigned URL generation
    """
    
    def __init__(
        self,
        endpoint: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        secure: Optional[bool] = None,
    ):
        """
        Initialize MinIO client.
        
        Args:
            endpoint: MinIO endpoint (default from settings)
            access_key: Access key (default from settings)
            secret_key: Secret key (default from settings)
            secure: Use HTTPS (default from settings)
        """
        self.endpoint = endpoint or settings.minio.endpoint
        self.access_key = access_key or settings.minio.access_key
        self.secret_key = secret_key or settings.minio.secret_key
        self.secure = secure if secure is not None else settings.minio.secure
        
        self._client = Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure,
        )
        
        logger.info(
            "MinioStorage initialized",
            endpoint=self.endpoint,
            secure=self.secure,
        )
    
    async def ensure_bucket(self, bucket_name: str) -> bool:
        """
        Ensure a bucket exists, creating if necessary.
        
        Args:
            bucket_name: Name of the bucket
            
        Returns:
            bool: True if bucket was created, False if already existed
        """
        loop = asyncio.get_event_loop()
        
        try:
            exists = await loop.run_in_executor(
                None, self._client.bucket_exists, bucket_name
            )
            
            if not exists:
                await loop.run_in_executor(
                    None, self._client.make_bucket, bucket_name
                )
                logger.info("Created bucket", bucket=bucket_name)
                return True
            
            return False
            
        except S3Error as e:
            logger.error("Bucket operation failed", bucket=bucket_name, error=str(e))
            raise
    
    async def upload_file(
        self,
        bucket_name: str,
        object_name: str,
        file_path: Path,
        content_type: str = "application/octet-stream",
        metadata: Optional[dict[str, str]] = None,
    ) -> dict:
        """
        Upload a file to storage.
        
        Args:
            bucket_name: Target bucket
            object_name: Object key/path
            file_path: Local file path
            content_type: MIME type
            metadata: Optional metadata
            
        Returns:
            dict: Upload result with etag and version_id
        """
        await self.ensure_bucket(bucket_name)
        loop = asyncio.get_event_loop()
        
        try:
            result = await loop.run_in_executor(
                None,
                lambda: self._client.fput_object(
                    bucket_name,
                    object_name,
                    str(file_path),
                    content_type=content_type,
                    metadata=metadata,
                ),
            )
            
            logger.debug(
                "Uploaded file",
                bucket=bucket_name,
                object=object_name,
                etag=result.etag,
            )
            
            return {
                "etag": result.etag,
                "version_id": result.version_id,
                "bucket": bucket_name,
                "object": object_name,
            }
            
        except S3Error as e:
            logger.error(
                "Upload failed",
                bucket=bucket_name,
                object=object_name,
                error=str(e),
            )
            raise
    
    async def upload_data(
        self,
        bucket_name: str,
        object_name: str,
        data: bytes | BinaryIO,
        length: int = -1,
        content_type: str = "application/octet-stream",
        metadata: Optional[dict[str, str]] = None,
    ) -> dict:
        """
        Upload data directly to storage.
        
        Args:
            bucket_name: Target bucket
            object_name: Object key/path
            data: Bytes or file-like object
            length: Data length (-1 for unknown)
            content_type: MIME type
            metadata: Optional metadata
            
        Returns:
            dict: Upload result
        """
        await self.ensure_bucket(bucket_name)
        loop = asyncio.get_event_loop()
        
        if isinstance(data, bytes):
            data = BytesIO(data)
            length = len(data.getvalue())
        
        try:
            result = await loop.run_in_executor(
                None,
                lambda: self._client.put_object(
                    bucket_name,
                    object_name,
                    data,
                    length,
                    content_type=content_type,
                    metadata=metadata,
                ),
            )
            
            return {
                "etag": result.etag,
                "version_id": result.version_id,
                "bucket": bucket_name,
                "object": object_name,
            }
            
        except S3Error as e:
            logger.error(
                "Upload failed",
                bucket=bucket_name,
                object=object_name,
                error=str(e),
            )
            raise
    
    async def download_file(
        self,
        bucket_name: str,
        object_name: str,
        file_path: Path,
    ) -> Path:
        """
        Download an object to a file.
        
        Args:
            bucket_name: Source bucket
            object_name: Object key/path
            file_path: Local destination path
            
        Returns:
            Path: Downloaded file path
        """
        loop = asyncio.get_event_loop()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            await loop.run_in_executor(
                None,
                lambda: self._client.fget_object(
                    bucket_name,
                    object_name,
                    str(file_path),
                ),
            )
            
            logger.debug(
                "Downloaded file",
                bucket=bucket_name,
                object=object_name,
                path=str(file_path),
            )
            
            return file_path
            
        except S3Error as e:
            logger.error(
                "Download failed",
                bucket=bucket_name,
                object=object_name,
                error=str(e),
            )
            raise
    
    async def download_data(
        self,
        bucket_name: str,
        object_name: str,
    ) -> bytes:
        """
        Download an object as bytes.
        
        Args:
            bucket_name: Source bucket
            object_name: Object key/path
            
        Returns:
            bytes: Object data
        """
        loop = asyncio.get_event_loop()
        
        try:
            response = await loop.run_in_executor(
                None,
                lambda: self._client.get_object(bucket_name, object_name),
            )
            
            data = response.read()
            response.close()
            response.release_conn()
            
            return data
            
        except S3Error as e:
            logger.error(
                "Download failed",
                bucket=bucket_name,
                object=object_name,
                error=str(e),
            )
            raise
    
    async def list_objects(
        self,
        bucket_name: str,
        prefix: str = "",
        recursive: bool = True,
    ) -> list[dict]:
        """
        List objects in a bucket.
        
        Args:
            bucket_name: Bucket to list
            prefix: Path prefix filter
            recursive: Include subdirectories
            
        Returns:
            list[dict]: Object metadata list
        """
        loop = asyncio.get_event_loop()
        
        try:
            objects = await loop.run_in_executor(
                None,
                lambda: list(self._client.list_objects(
                    bucket_name,
                    prefix=prefix,
                    recursive=recursive,
                )),
            )
            
            return [
                {
                    "name": obj.object_name,
                    "size": obj.size,
                    "etag": obj.etag,
                    "last_modified": obj.last_modified.isoformat() if obj.last_modified else None,
                    "is_dir": obj.is_dir,
                }
                for obj in objects
            ]
            
        except S3Error as e:
            logger.error(
                "List failed",
                bucket=bucket_name,
                prefix=prefix,
                error=str(e),
            )
            raise
    
    async def delete_object(
        self,
        bucket_name: str,
        object_name: str,
    ) -> None:
        """
        Delete an object.
        
        Args:
            bucket_name: Bucket name
            object_name: Object to delete
        """
        loop = asyncio.get_event_loop()
        
        try:
            await loop.run_in_executor(
                None,
                lambda: self._client.remove_object(bucket_name, object_name),
            )
            
            logger.debug(
                "Deleted object",
                bucket=bucket_name,
                object=object_name,
            )
            
        except S3Error as e:
            logger.error(
                "Delete failed",
                bucket=bucket_name,
                object=object_name,
                error=str(e),
            )
            raise
    
    async def get_presigned_url(
        self,
        bucket_name: str,
        object_name: str,
        expires: int = 3600,
    ) -> str:
        """
        Generate a presigned URL for download.
        
        Args:
            bucket_name: Bucket name
            object_name: Object name
            expires: URL expiration in seconds
            
        Returns:
            str: Presigned URL
        """
        loop = asyncio.get_event_loop()
        from datetime import timedelta
        
        try:
            url = await loop.run_in_executor(
                None,
                lambda: self._client.presigned_get_object(
                    bucket_name,
                    object_name,
                    expires=timedelta(seconds=expires),
                ),
            )
            
            return url
            
        except S3Error as e:
            logger.error(
                "Presigned URL generation failed",
                bucket=bucket_name,
                object=object_name,
                error=str(e),
            )
            raise
    
    async def stat_object(
        self,
        bucket_name: str,
        object_name: str,
    ) -> dict:
        """
        Get object metadata.
        
        Args:
            bucket_name: Bucket name
            object_name: Object name
            
        Returns:
            dict: Object metadata
        """
        loop = asyncio.get_event_loop()
        
        try:
            stat = await loop.run_in_executor(
                None,
                lambda: self._client.stat_object(bucket_name, object_name),
            )
            
            return {
                "size": stat.size,
                "etag": stat.etag,
                "content_type": stat.content_type,
                "last_modified": stat.last_modified.isoformat() if stat.last_modified else None,
                "metadata": dict(stat.metadata) if stat.metadata else {},
            }
            
        except S3Error as e:
            logger.error(
                "Stat failed",
                bucket=bucket_name,
                object=object_name,
                error=str(e),
            )
            raise
