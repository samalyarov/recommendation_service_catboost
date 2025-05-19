import os
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from main import app, get_db, Base

# Test database URL
TEST_DATABASE_URL = "sqlite:///:memory:"

@pytest.fixture(scope="session")
def test_db_engine():
    """Create a test database engine."""
    engine = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    return engine

@pytest.fixture(scope="function")
def db_session(test_db_engine):
    """Create a fresh database session for each test."""
    TestingSessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=test_db_engine
    )
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()

@pytest.fixture(scope="function")
def client(db_session):
    """Create a test client with the test database session."""
    def override_get_db():
        try:
            yield db_session
        finally:
            db_session.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()

@pytest.fixture(scope="session")
def test_user_data():
    """Sample user data for testing."""
    return {
        "user_id": 1,
        "gender": "M",
        "age": 25,
        "country": "Russia",
        "city": "Moscow",
        "exp_group": "control",
        "os": "iOS",
        "source": "organic",
        "unique_post_interactions": 10,
        "posts_liked": 5,
        "total_views": 100,
        "posts_liked_share": 0.5
    }

@pytest.fixture(scope="session")
def test_post_data():
    """Sample post data for testing."""
    return {
        "post_id": 1,
        "text": "Test post content",
        "topic": "technology",
        "post_length": 17,
        "unique_user_interactions": 5,
        "user_likes": 3,
        "total_post_views": 20,
        "post_likability": 0.6
    } 