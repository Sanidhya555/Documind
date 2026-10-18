"""Synchronous adapter for the offline evaluation runner.

FastAPI routes should call the asynchronous parser directly.
"""
import asyncio
import os

from utils.parser import extract_document_content


class DoclingLoader:
    def load(self, file_path: str) -> str:
        return asyncio.run(extract_document_content(file_path))


class IngestionPipeline:
    def __init__(self):
        extensions = (
            ".pdf",
            ".docx",
            ".xlsx",
            ".pptx",
            ".png",
            ".jpg",
            ".jpeg",
            ".tif",
            ".tiff",
            ".bmp",
            ".webp",
        )
        self.loaders = {extension: DoclingLoader() for extension in extensions}

    def process(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.loaders:
            raise ValueError(f"Unsupported file format: {ext}")
        return self.loaders[ext].load(file_path)
