# ============================================================
# app/schemas.py
# PURPOSE: Define data shapes for API input and output
# Pydantic validates all data automatically
# ============================================================

from pydantic import BaseModel
# BaseModel: base class for all our schemas
# All schema classes inherit from this
# Pydantic automatically validates data types
from typing import Optional, List
# List: for lists of items
# Optional: means the field can be None
# ============================================================
# WHAT IS PYDANTIC?
# ============================================================

# Pydantic is a data validation library.
#
# WITHOUT Pydantic:
# User sends amount = "hello" (string instead of number)
# Your code crashes or behaves unexpectedly
#
# WITH Pydantic:
# User sends amount = "hello"
# Pydantic automatically returns clear error:
# "amount must be a number"
# Your code never runs with bad data
#
# FastAPI uses Pydantic automatically for all schemas.
# It also generates API documentation from these schemas.


# ============================================================
# SINGLE EXPENSE SCHEMA (for API responses)
# ============================================================
class ExpenseResponse(BaseModel):
    """
    Shape of a single expense when returned by API.
    Used in GET endpoints to show expense data .
    """
    id:int
    data:str
    description: str
    amount:float
    category:str
    confidence:Optional[float] = None
    prediction_method : Optional[str] = None
    upload_batch : Optional[str] = None

    class Config:
        #Allows Pydantic to read data from
        # SQLAlchemy objects directly
        # without this it only reads plain dicts
        from_attributes = True

# SUMMARY SCHEMAS

class CategorySummary(BaseModel):

    """
    Spending summary for one category.
    Example: Food → total 5400.00, count 12 expenses
    """
    category : str
    total_amount : float
    expense_count : int
    percentage : float

class SpendingSummary(BaseModel):
    """
    Complete spending summary returned by /summary endpoint.
    Contains overview and breakdown by category.
    """

    total_expenses: int          # number of expense records
    total_amount: float          # sum of all amounts
    average_expense: float       # average per transaction
    highest_amount: float        # most expensive transaction
    lowest_amount: float         # cheapest transaction
    categories: List[CategorySummary]  # breakdown per category
    highest_category: str        # category with most spending
    date_range_start: str        # earliest expense date
    date_range_end: str          # latest expense date

# ============================================================
# INSIGHTS SCHEMA
# ============================================================

class InsightItem(BaseModel):
    """A single insight about spending"""

    insight_type: str   # "warning", "info", "tip"
    message: str        # the actual insight text


class InsightsResponse(BaseModel):
    """All insights returned by /insights endpoint"""

    insights: List[InsightItem]
    total_insights: int


# ============================================================
# UPLOAD RESPONSE SCHEMA
# ============================================================

class UploadResponse(BaseModel):
    """
    Response returned after user uploads an Excel file.
    Tells user what happened after upload.
    """

    message: str                    # success message
    upload_batch: str               # unique ID for this upload
    total_rows: int                 # how many expenses processed
    categories_found: List[str]     # which categories were found
    preview: List[ExpenseResponse]  # first 5 expenses as preview


# ============================================================
# ERROR SCHEMA
# ============================================================

class ErrorResponse(BaseModel):
    """Standard error response shape"""

    error: str    # error type
    detail: str   # detailed error message