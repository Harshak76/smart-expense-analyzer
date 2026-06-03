# ============================================================
# app/utils.py
# PURPOSE: Core business logic of our application
# Connects Excel → Pandas → ML Model → Database
# ============================================================

import pandas as pd
# pandas: read and clean Excel files
# industry standard for data manipulation

import uuid
# uuid: generates unique IDs
# We use it to create unique upload batch IDs
# uuid4() generates a random unique string like:
# "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

import os
# os: file path operations

from datetime import datetime
# datetime: for date operations and formatting

from sqlalchemy.orm import Session
# Session: type hint for database session parameter

from .models import Expense
# Import our Expense table class

from ml.predict import (
    load_model,
    predict_categories_bulk
)
# Import our ML prediction functions


# ============================================================
# LOAD ML MODEL ONCE WHEN MODULE IS IMPORTED
# ============================================================

# We load the model HERE at module level
# This means it loads ONCE when the app starts
# NOT every time a user uploads a file
# Loading a model takes time, so we do it once
# and reuse it for every prediction

print("Loading ML model...")
ml_model, keyword_rules = load_model()
print("ML model loaded successfully!")


# ============================================================
# STEP 1: READ AND VALIDATE EXCEL FILE
# ============================================================

def read_excel_file(file_path: str) -> pd.DataFrame:
    """
    Read uploaded Excel file into a pandas DataFrame.
    Validate that required columns exist.

    REQUIRED COLUMNS (flexible naming):
    → Date column: 'date', 'Date', 'DATE', 'transaction date'
    → Description: 'description', 'Description', 'desc', 'narration'
    → Amount: 'amount', 'Amount', 'AMOUNT', 'debit', 'credit'

    Returns cleaned DataFrame with standardized column names.
    """

    # Read Excel file into DataFrame
    # DataFrame = table with rows and columns (like Excel in Python)
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        raise ValueError(f"Could not read Excel file: {str(e)}")

    # Convert all column names to lowercase for easy matching
    df.columns = df.columns.str.lower().str.strip()

    print(f"File read successfully. Columns found: {list(df.columns)}")
    print(f"Total rows: {len(df)}")

    # ── FIND DATE COLUMN ──────────────────────────────────
    # User might name it differently, we check multiple options
    date_options = [
        'date', 'transaction date', 'txn date',
        'value date', 'posting date', 'trans date'
    ]
    date_col = None
    for option in date_options:
        if option in df.columns:
            date_col = option
            break

    if date_col is None:
        raise ValueError(
            "Date column not found. "
            "Please ensure your Excel file has a column named: "
            "'date' or 'transaction date'"
        )

    # ── FIND DESCRIPTION COLUMN ───────────────────────────
    desc_options = [
        'description', 'desc', 'narration', 'particulars',
        'details', 'remarks', 'transaction details', 'note'
    ]
    desc_col = None
    for option in desc_options:
        if option in df.columns:
            desc_col = option
            break

    if desc_col is None:
        raise ValueError(
            "Description column not found. "
            "Please ensure your Excel file has a column named: "
            "'description' or 'narration' or 'particulars'"
        )

    # ── FIND AMOUNT COLUMN ────────────────────────────────
    amount_options = [
        'amount', 'debit', 'credit', 'transaction amount',
        'txn amount', 'withdrawal', 'expense', 'value'
    ]
    amount_col = None
    for option in amount_options:
        if option in df.columns:
            amount_col = option
            break

    if amount_col is None:
        raise ValueError(
            "Amount column not found. "
            "Please ensure your Excel file has a column named: "
            "'amount' or 'debit' or 'credit'"
        )

    # ── STANDARDIZE COLUMN NAMES ──────────────────────────
    # Rename found columns to our standard names
    # So rest of code always uses same column names
    df = df.rename(columns={
        date_col: 'date',
        desc_col: 'description',
        amount_col: 'amount'
    })

    # Keep only the columns we need
    df = df[['date', 'description', 'amount']].copy()

    return df


# ============================================================
# STEP 2: CLEAN THE DATA
# ============================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize the expense data.

    Handles:
    → Missing values
    → Amount formatting (remove Rs., $, commas)
    → Date standardization
    → Description cleaning
    → Duplicate removal
    """

    original_count = len(df)

    # ── CLEAN DESCRIPTION ─────────────────────────────────

    # Convert to string (in case some values are numbers)
    df['description'] = df['description'].astype(str)

    # Strip extra whitespace from both ends
    df['description'] = df['description'].str.strip()

    # Replace multiple spaces with single space
    df['description'] = df['description'].str.replace(
        r'\s+', ' ', regex=True
    )

    # Remove rows where description is empty or 'nan'
    df = df[df['description'].str.lower() != 'nan']
    df = df[df['description'] != '']
    df = df[df['description'].str.len() > 1]

    # ── CLEAN AMOUNT ──────────────────────────────────────

    # Convert to string first for cleaning
    df['amount'] = df['amount'].astype(str)

    # Remove currency symbols and formatting
    # Rs., INR, $, £, €, commas, spaces
    df['amount'] = df['amount'].str.replace(
        r'[Rs\.INR$£€,\s]', '', regex=True
    )

    # Convert to float
    # errors='coerce' turns invalid values into NaN
    # instead of crashing
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')

    # Take absolute value (make negatives positive)
    # Bank statements sometimes show debits as negative
    df['amount'] = df['amount'].abs()

    # Remove rows where amount is NaN or zero
    df = df.dropna(subset=['amount'])
    df = df[df['amount'] > 0]

    # ── CLEAN DATE ────────────────────────────────────────

    # Convert to string first
    df['date'] = df['date'].astype(str)

    # Strip whitespace
    df['date'] = df['date'].str.strip()

    # Try to parse dates flexibly
    # infer_datetime_format tries multiple formats automatically
    df['date'] = pd.to_datetime(
    df['date'],
    errors='coerce'
    # pandas automatically detects date format
    # infer_datetime_format was removed in newer pandas
    # it now infers format automatically by default
    )

    # Remove rows with invalid dates
    df = df.dropna(subset=['date'])

    # Standardize date format to YYYY-MM-DD string
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')

    # ── REMOVE DUPLICATES ─────────────────────────────────

    # Remove exact duplicate rows
    df = df.drop_duplicates(
        subset=['date', 'description', 'amount']
    )

    # Reset index after all the row removals
    # So index goes 0, 1, 2, 3... again
    df = df.reset_index(drop=True)

    cleaned_count = len(df)
    removed = original_count - cleaned_count

    print(f"Data cleaning complete.")
    print(f"Original rows: {original_count}")
    print(f"Cleaned rows: {cleaned_count}")
    print(f"Removed rows: {removed} (empty/invalid/duplicate)")

    return df


# ============================================================
# STEP 3: CATEGORIZE EXPENSES USING ML
# ============================================================

def categorize_expenses(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run our hybrid ML system on every expense row.
    Adds 'category', 'confidence', 'prediction_method' columns.
    """

    print(f"Categorizing {len(df)} expenses...")

    # predict_categories_bulk processes all rows at once
    # Returns same df with new columns added
    df = predict_categories_bulk(df, ml_model, keyword_rules)

    print("Categorization complete!")
    print("Category distribution:")
    print(df['category'].value_counts())

    return df


# ============================================================
# STEP 4: SAVE TO DATABASE
# ============================================================

def save_to_database(
    df: pd.DataFrame,
    db: Session
) -> str:
    """
    Save all categorized expenses to SQLite database.

    Returns the upload_batch ID so user can reference
    this specific upload later.
    """

    # Generate unique batch ID for this upload
    # Every upload gets its own unique identifier
    # uuid4() generates something like: "a1b2c3d4-e5f6..."
    # str() converts it to a readable string
    upload_batch = str(uuid.uuid4())

    print(f"Saving {len(df)} expenses to database...")
    print(f"Upload batch ID: {upload_batch}")

    # Convert each row to an Expense object and save
    expenses_to_save = []

    for _, row in df.iterrows():
        # iterrows() goes through DataFrame row by row
        # _ is the index (we ignore it with _)
        # row is a Series with all column values

        expense = Expense(
            date=str(row['date']),
            description=str(row['description']),
            amount=float(row['amount']),
            category=str(row['category']),
            confidence=float(row.get('confidence', 0.0)),
            prediction_method=str(
                row.get('prediction_method', 'rule_based')
            ),
            upload_batch=upload_batch
        )
        expenses_to_save.append(expense)

    # Add all at once (more efficient than one by one)
    db.add_all(expenses_to_save)
    db.commit()

    print(f"All expenses saved successfully!")

    return upload_batch


# ============================================================
# STEP 5: GENERATE SPENDING SUMMARY
# ============================================================

def generate_summary(upload_batch: str, db: Session) -> dict:
    """
    Generate spending summary for a specific upload batch.

    Calculates:
    → Total expenses and amount
    → Average, highest, lowest amounts
    → Breakdown by category with percentages
    → Date range of expenses
    → Highest spending category
    """

    # Fetch all expenses for this batch
    expenses = db.query(Expense).filter(
        Expense.upload_batch == upload_batch
    ).all()

    if not expenses:
        return {}

    # Extract amounts as a list for calculations
    amounts = [exp.amount for exp in expenses]
    total_amount = sum(amounts)

    # ── CATEGORY BREAKDOWN ────────────────────────────────

    # Group expenses by category
    # Using a dictionary to accumulate totals
    category_data = {}

    for exp in expenses:
        cat = exp.category
        if cat not in category_data:
            category_data[cat] = {
                'total': 0.0,
                'count': 0
            }
        category_data[cat]['total'] += exp.amount
        category_data[cat]['count'] += 1

    # Build category summary list
    categories = []
    for cat, data in category_data.items():
        percentage = round(
            (data['total'] / total_amount) * 100, 2
        )
        categories.append({
            'category': cat,
            'total_amount': round(data['total'], 2),
            'expense_count': data['count'],
            'percentage': percentage
        })

    # Sort categories by total amount (highest first)
    categories.sort(
        key=lambda x: x['total_amount'],
        reverse=True
    )

    # Find highest spending category
    highest_category = categories[0]['category'] if categories else 'None'

    # ── DATE RANGE ────────────────────────────────────────
    dates = [exp.date for exp in expenses]
    dates.sort()
    date_start = dates[0]
    date_end = dates[-1]

    # ── BUILD SUMMARY ─────────────────────────────────────
    summary = {
        'total_expenses': len(expenses),
        'total_amount': round(total_amount, 2),
        'average_expense': round(total_amount / len(expenses), 2),
        'highest_amount': round(max(amounts), 2),
        'lowest_amount': round(min(amounts), 2),
        'categories': categories,
        'highest_category': highest_category,
        'date_range_start': date_start,
        'date_range_end': date_end
    }

    return summary


# ============================================================
# STEP 6: GENERATE INSIGHTS
# ============================================================

def generate_insights(upload_batch: str, db: Session) -> list:
    """
    Generate smart insights about spending patterns.

    Returns a list of insight dictionaries with:
    → insight_type: "warning", "info", "tip"
    → message: the actual insight text
    """

    expenses = db.query(Expense).filter(
        Expense.upload_batch == upload_batch
    ).all()

    if not expenses:
        return []

    insights = []
    amounts = [exp.amount for exp in expenses]
    total_amount = sum(amounts)

    # Calculate category totals and percentages
    category_totals = {}
    for exp in expenses:
        cat = exp.category
        if cat not in category_totals:
            category_totals[cat] = 0.0
        category_totals[cat] += exp.amount

    category_percentages = {
        cat: (total / total_amount) * 100
        for cat, total in category_totals.items()
    }

    # ── INSIGHT 1: High Food Spending ─────────────────────
    food_pct = category_percentages.get('Food', 0)
    if food_pct > 30:
        insights.append({
            'insight_type': 'warning',
            'message': (
                f"You spent {food_pct:.1f}% of your budget on Food. "
                f"Consider meal prepping to reduce dining expenses."
            )
        })

    # ── INSIGHT 2: No Savings/Income ──────────────────────
    if 'Income' not in category_totals:
        insights.append({
            'insight_type': 'tip',
            'message': (
                "No income records found this month. "
                "Consider tracking your income alongside expenses "
                "for a complete financial picture."
            )
        })

    # ── INSIGHT 3: High Bills Percentage ──────────────────
    bills_pct = category_percentages.get('Bills', 0)
    if bills_pct > 40:
        insights.append({
            'insight_type': 'warning',
            'message': (
                f"Bills account for {bills_pct:.1f}% of spending. "
                f"Review your subscriptions and utilities "
                f"for potential savings."
            )
        })

    # ── INSIGHT 4: Highest Single Expense ─────────────────
    max_expense = max(expenses, key=lambda x: x.amount)
    insights.append({
        'insight_type': 'info',
        'message': (
            f"Your highest single expense was "
            f"Rs.{max_expense.amount:.2f} "
            f"on '{max_expense.description}' "
            f"categorized as {max_expense.category}."
        )
    })

    # ── INSIGHT 5: Entertainment Warning ──────────────────
    ent_pct = category_percentages.get('Entertainment', 0)
    if ent_pct > 20:
        insights.append({
            'insight_type': 'warning',
            'message': (
                f"Entertainment spending is at {ent_pct:.1f}%. "
                f"Check for unused streaming subscriptions."
            )
        })

    # ── INSIGHT 6: Good Health Investment ─────────────────
    health_pct = category_percentages.get('Health', 0)
    if health_pct > 0 and health_pct < 15:
        insights.append({
            'insight_type': 'info',
            'message': (
                f"You spent {health_pct:.1f}% on Health. "
                f"Maintaining health investments is a good habit."
            )
        })

    # ── INSIGHT 7: Total Spending Summary ─────────────────
    insights.append({
        'insight_type': 'info',
        'message': (
            f"Total spending analyzed: "
            f"Rs.{total_amount:.2f} across "
            f"{len(expenses)} transactions in "
            f"{len(category_totals)} categories."
        )
    })

    return insights


# ============================================================
# MASTER FUNCTION: Process entire uploaded file
# ============================================================

def process_expense_file(file_path: str, db: Session) -> dict:
    """
    Master function that runs all steps in order.
    Called by our FastAPI upload endpoint.

    FLOW:
    Excel file path
    → read_excel_file()
    → clean_data()
    → categorize_expenses()
    → save_to_database()
    → return upload_batch + preview

    Returns dict with upload_batch and preview data.
    """

    print()
    print("=" * 50)
    print("PROCESSING EXPENSE FILE")
    print("=" * 50)

    # Step 1: Read Excel file
    df = read_excel_file(file_path)

    # Step 2: Clean data
    df = clean_data(df)

    # Check if any valid data remains after cleaning
    if len(df) == 0:
        raise ValueError(
            "No valid expense data found after cleaning. "
            "Please check your Excel file format."
        )

    # Step 3: Categorize using ML
    df = categorize_expenses(df)

    # Step 4: Save to database
    upload_batch = save_to_database(df, db)

    # Step 5: Build preview (first 5 rows)
    preview = df.head(5).to_dict(orient='records')

    print("=" * 50)
    print("FILE PROCESSING COMPLETE")
    print("=" * 50)
    print()

    return {
        'upload_batch': upload_batch,
        'total_rows': len(df),
        'categories_found': df['category'].unique().tolist(),
        'preview': preview
    }