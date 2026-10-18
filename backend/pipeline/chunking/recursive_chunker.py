from typing import List

class RecursiveChunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> List[str]:
        chunks, start = [], 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if end < len(text):
                # Prefer a whitespace boundary so a token is not split across
                # adjacent chunks. If a single token exceeds the configured
                # size, retain the hard bound and split that token as a last resort.
                boundary = text.rfind(" ", start + 1, end + 1)
                if boundary > start:
                    end = boundary
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            next_start = max(end - self.overlap, start + 1)
            while next_start < len(text) and text[next_start].isspace():
                next_start += 1
            start = next_start
        return chunks
