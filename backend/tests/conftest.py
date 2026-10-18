"""
pytest configuration for backend tests.

Sets PYTHONPATH so imports resolve from the backend/ root.
"""
import sys
import os

# Add backend root to PYTHONPATH so 'pipeline.*' and 'app.*' resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set safe test environment variables before any app imports
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-testing-only")
os.environ.setdefault("ADMIN_USERNAME", "admin")
os.environ.setdefault("ADMIN_PASSWORD", "admin123")
os.environ.setdefault("USE_OLLAMA", "false")
os.environ.setdefault("ADAPTIVE_RETRIEVAL_ENABLED", "true")
os.environ.setdefault("LAYA_ENABLED", "false")
os.environ.setdefault("ALLOW_MODEL_DOWNLOADS", "false")
os.environ.setdefault("LOG_LEVEL", "WARNING")  # suppress logs during tests
