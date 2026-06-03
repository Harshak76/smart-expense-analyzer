from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

# DATABASE URL

DATABASE_URL = "sqlite:///./expenses.db"

# CREATE ENGINE

engine = create_engine(
    DATABASE_URL,
    connect_args = {"check_same_thread": False}
)

# CREATE SESSION FACTORY

SessionLocal = sessionmaker(
    autocommit = False,
    autoflush = False,
    bind = engine
)

Base = declarative_base()

# DATABASE SESSION DEPENDENCY

def get_db():
    """
    Gives FastAPI a database session.
    Used as a DEPENDENCY in our API endpoints.

    Code before yield runs BEFORE the endpoint.
    Code after yield runs AFTER the endpoint.
    Session is ALWAYS closed even if error occurs.
    """
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


