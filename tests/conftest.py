"""
Pytest configuration and test environment isolation.
Redirects SQLite database and raw file storage to an isolated temporary sandbox
so test executions NEVER touch or pollute production data/graphrag.db or data/raw.
"""

import pytest
import asyncio
from pathlib import Path
from backend.config import get_settings
from backend.database import init_db
import backend.database as db_module


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment(tmp_path_factory):
    """
    Isolate tests completely from production database and storage.
    Creates a temporary directory for test databases and uploaded files.
    """
    temp_dir = tmp_path_factory.mktemp("test_sandbox")
    test_db_path = temp_dir / "test_graphrag.db"
    test_data_dir = temp_dir / "raw"
    test_data_dir.mkdir(parents=True, exist_ok=True)

    settings = get_settings()
    original_db_path = settings.db_path
    original_data_dir = settings.data_dir
    original_global_db_path = db_module.DB_PATH

    # Redirect paths to temporary sandbox
    settings.db_path = test_db_path
    settings.data_dir = test_data_dir

    # Initialize the SQLite schema in the test sandbox
    asyncio.run(init_db(test_db_path))

    yield

    # Restore original settings
    settings.db_path = original_db_path
    settings.data_dir = original_data_dir
    if original_global_db_path:
        asyncio.run(init_db(original_global_db_path))
