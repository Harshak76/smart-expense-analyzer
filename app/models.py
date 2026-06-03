# ============================================================
# app/models.py
# PURPOSE : Define databse table structure
# Echa class here =  one table in the database 
# Each class attribute = one column in that table 
# ============================================================

from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func
from .database import Base

class Expense(Base):
    """
    This class defines our 'expenses' table.
    Every instance of this class = one row in the table
    """

    __tablename__ = "expenses"

    id = Column(
        Integer,
        primary_key=True,   # unique identifier for each row
        index=True,         # creates index for fast lookup
        autoincrement=True  # automatically assigns 1, 2, 3...
    )

    date = Column(
        String,
        nullable=False  # REQUIRED, cannot be empty
    )
    # Store date as string: "2024-01-15"

    description = Column(
        String,
        nullable=False
    )
    # Expense description from Excel file
    # Example: "mcdonalds burger", "uber cab"

    amount = Column(
        Float,
        nullable=False
    )
    # Expense amount, allows decimals like 250.50

    category = Column(
        String,
        nullable=False
    )
    # Category assigned by our hybrid ML system
    # Example: "Food", "Transport", "Bills"

    confidence = Column(
        Float,
        nullable=True,
        default=0.0
    )
    # How confident our model was (0 to 100)

    prediction_method = Column(
        String,
        nullable=True,
        default="rule_based"
    )
    # "rule_based" or "ml_model"

    created_at = Column(
        DateTime,
        server_default=func.now()
        # database sets this automatically
        # inserts current timestamp
    )
    # When this record was added to database

    upload_batch = Column(
        String,
        nullable=True
    )
    # Groups expenses from same upload together
    # All rows from one file share same batch id

    def to_dict(self):
        """
        Convert database object to plain dictionary.
        FastAPI returns JSON which cannot contain
        SQLAlchemy objects, so we convert to dict first.
        """
        return {
            "id": self.id,
            "date": self.date,
            "description": self.description,
            "amount": self.amount,
            "category": self.category,
            "confidence": self.confidence,
            "prediction_method": self.prediction_method,
            "created_at": str(self.created_at),
            "upload_batch": self.upload_batch
        }

    def __repr__(self):
        """String representation for debugging"""
        return (
            f"<Expense("
            f"id={self.id}, "
            f"description='{self.description}', "
            f"amount={self.amount}, "
            f"category='{self.category}'"
            f")>"
        )

