# ============================================================
# ml/predict.py
# PURPOSE: Load saved hybrid model and make predictions
# Used by our FastAPI to categorize expenses
# ============================================================

import pickle
import os
import pandas as pd


# ============================================================
# LOAD THE SAVED MODEL
# ============================================================

def load_model():
    """
    Load trained model and keyword rules from saved file.
    Returns both ml_model and keyword_rules.
    """
    
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, 'model.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Model not found. Please run ml/train_model.py first"
        )
    
    with open(model_path, 'rb') as file:
        saved_object = pickle.load(file)
    
    ml_model = saved_object['ml_model']
    keyword_rules = saved_object['keyword_rules']
    
    return ml_model, keyword_rules


# ============================================================
# RULE BASED PREDICTION
# ============================================================

def rule_based_predict(description: str, keyword_rules: dict):
    """Try to categorize using keyword matching"""
    
    text = description.lower().strip()
    matches = {}
    
    for category, keywords in keyword_rules.items():
        match_count = sum(1 for kw in keywords if kw in text)
        if match_count > 0:
            matches[category] = match_count
    
    if not matches:
        return None
    
    best_category = max(matches, key=matches.get)
    total_matches = sum(matches.values())
    confidence = round(
        (matches[best_category] / total_matches) * 100, 2
    )
    confidence = max(confidence, 85.0)
    
    return {
        'category': best_category,
        'confidence': confidence,
        'method': 'rule_based'
    }


# ============================================================
# HYBRID PREDICTION - SINGLE EXPENSE
# ============================================================

def predict_category(description: str, ml_model, keyword_rules) -> dict:
    """
    Predict category for a single expense description.
    Uses rules first, ML as backup.
    
    INPUT:  description = "mcdonalds burger"
    OUTPUT: {
                category: "Food",
                confidence: 95.0,
                method: "rule_based"
            }
    """
    
    if not description or not description.strip():
        return {
            'category': 'Other',
            'confidence': 0.0,
            'method': 'default'
        }
    
    # Try rule based first
    rule_result = rule_based_predict(description, keyword_rules)
    
    if rule_result is not None:
        return rule_result
    
    # ML fallback
    cleaned = description.lower().strip()
    cleaned = ''.join(
        c for c in cleaned if c.isalpha() or c == ' '
    )
    
    category = ml_model.predict([cleaned])[0]
    probabilities = ml_model.predict_proba([cleaned])[0]
    confidence = round(max(probabilities) * 100, 2)
    
    return {
        'category': category,
        'confidence': confidence,
        'method': 'ml_model'
    }


# ============================================================
# HYBRID PREDICTION - BULK (entire DataFrame)
# ============================================================

def predict_categories_bulk(
    df: pd.DataFrame,
    ml_model,
    keyword_rules
) -> pd.DataFrame:
    """
    Predict categories for ALL expenses in a DataFrame.
    
    INPUT:  DataFrame with 'description' column
    OUTPUT: Same DataFrame with these new columns added:
            - category
            - confidence
            - prediction_method
    """
    
    categories = []
    confidences = []
    methods = []
    
    for description in df['description']:
        result = predict_category(
            str(description),
            ml_model,
            keyword_rules
        )
        categories.append(result['category'])
        confidences.append(result['confidence'])
        methods.append(result['method'])
    
    df['category'] = categories
    df['confidence'] = confidences
    df['prediction_method'] = methods
    
    return df