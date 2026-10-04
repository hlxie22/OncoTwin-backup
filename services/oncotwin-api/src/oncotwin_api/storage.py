from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path

import boto3

from .config import get_settings


class ObjectStore:
    def put_bytes(self, key: str, data: bytes, content_type: str | None = None) -> None:
        raise NotImplementedError

    def get_bytes(self, key: str) -> bytes:
        raise NotImplementedError

    def delete(self, key: str) -> None:
        raise NotImplementedError


class LocalObjectStore(ObjectStore):
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if self.root not in p.parents and p != self.root:
            raise ValueError("unsafe storage key")
        return p

    def put_bytes(self, key: str, data: bytes, content_type: str | None = None) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def get_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def delete(self, key: str) -> None:
        path = self._path(key)
        if path.exists():
            path.unlink()


class S3ObjectStore(ObjectStore):
    def __init__(self):
        settings = get_settings()
        if not settings.s3_bucket:
            raise RuntimeError("ONCOTWIN_S3_BUCKET is required for s3 storage")
        self.bucket = settings.s3_bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint_url,
            region_name=settings.s3_region,
            aws_access_key_id=settings.s3_access_key_id,
            aws_secret_access_key=settings.s3_secret_access_key,
        )

    def put_bytes(self, key: str, data: bytes, content_type: str | None = None) -> None:
        kwargs = {"Bucket": self.bucket, "Key": key, "Body": io.BytesIO(data)}
        if content_type:
            kwargs["ContentType"] = content_type
        self.client.put_object(**kwargs)

    def get_bytes(self, key: str) -> bytes:
        return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)


@lru_cache(maxsize=1)
def get_object_store() -> ObjectStore:
    settings = get_settings()
    if settings.storage_backend == "s3":
        return S3ObjectStore()
    return LocalObjectStore(settings.storage_root)
