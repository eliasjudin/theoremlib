"""
Pytest configuration and shared fixtures
"""

import pytest
import os
import tempfile
import json
from pathlib import Path
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Set up test database URL
TEST_DB_URL = os.getenv("TEST_DATABASE_URL", "postgresql+asyncpg://test:test@localhost/test_theoremlib")

# Configure pytest to handle async tests
def pytest_configure(config):
    """Configure pytest with async plugin"""
    config.addinivalue_line("markers", "asyncio: mark test as async")

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def test_db():
    """Create test database and tables"""
    from ..database.models import Base
    
    # Create async engine
    engine = create_async_engine(TEST_DB_URL)
    
    async def init_db():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
    
    asyncio.run(init_db())
    return engine

@pytest.fixture
async def db_session(test_db):
    """Provide async database session for tests"""
    async_session = sessionmaker(
        test_db, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session
        await session.rollback()

@pytest.fixture(scope="session")
def test_storage():
    """Create temporary storage directory"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

@pytest.fixture(scope="session")
def test_ontology():
    """Create test ontology file"""
    ontology = {
        "geometry": {
            "name": "Geometry",
            "category": "geometry",
            "parent_concepts": [],
            "related_concepts": ["trigonometry"]
        },
        "trigonometry": {
            "name": "Trigonometry",
            "category": "geometry",
            "parent_concepts": ["geometry"],
            "related_concepts": ["geometry"]
        },
        "algebra": {
            "name": "Algebra",
            "category": "algebra",
            "parent_concepts": [],
            "related_concepts": ["linear_algebra"]
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(ontology, f)
        ontology_path = Path(f.name)
    
    yield ontology_path
    ontology_path.unlink()

@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration"""
    return {
        "STORAGE_PATH": str(test_storage),
        "ONTOLOGY_PATH": str(test_ontology),
        "OCR_API_KEY": "test_key",
        "DATABASE_URL": TEST_DB_URL,
        "ALLOWED_ORIGINS": ["http://localhost:3000"]
    }