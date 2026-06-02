# ============================================================
# ml/train_model.py
# PURPOSE: Train ML model using a hybrid approach
#          Rule based matching + ML for unknown cases
# ============================================================

import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# KEYWORD RULES
# These are our rule based patterns
# If a description matches these keywords
# we categorize it immediately without ML
# ============================================================

CATEGORY_KEYWORDS = {
    'Food': [
        'mcdonalds', 'burger king', 'pizza', 'dominos', 'kfc',
        'subway', 'starbucks', 'dunkin', 'restaurant', 'swiggy',
        'zomato', 'grocery', 'supermarket', 'vegetables', 'fruits',
        'bakery', 'cafe', 'food', 'meal', 'lunch', 'dinner',
        'breakfast', 'snack', 'coffee', 'tea', 'juice', 'milk',
        'bread', 'eggs', 'rice', 'dal', 'chicken', 'fish',
        'biryani', 'canteen', 'tiffin', 'kitchen', 'cook',
        'eating', 'dine', 'takeaway', 'delivery food'
    ],
    'Transport': [
        'uber', 'ola', 'auto', 'rickshaw', 'bus', 'train',
        'metro', 'petrol', 'diesel', 'fuel', 'parking', 'toll',
        'rapido', 'flight', 'airline', 'airport', 'cab',
        'taxi', 'irctc', 'makemytrip', 'redbus', 'goibibo',
        'vehicle', 'car wash', 'driving', 'fastag', 'highway',
        'ferry', 'transport', 'travel ticket', 'boat ticket',
        'bike repair', 'car service', 'tyre', 'engine',
        'motor', 'commute', 'railway', 'road trip'
    ],
    'Entertainment': [
        'netflix', 'spotify', 'amazon prime', 'youtube premium',
        'movie', 'cinema', 'theatre', 'concert', 'amusement',
        'hotstar', 'zee5', 'disney', 'sony liv', 'gaming',
        'playstation', 'xbox', 'steam', 'bowling', 'escape room',
        'karaoke', 'pub', 'arcade', 'virtual reality', 'museum',
        'zoo', 'safari', 'comedy show', 'live show', 'ipl',
        'stadium', 'swimming', 'golf', 'fun', 'leisure',
        'holiday package', 'tour package', 'sightseeing',
        'entertainment', 'subscription streaming'
    ],
    'Bills': [
        'electricity', 'water bill', 'internet bill', 'broadband',
        'recharge', 'postpaid', 'prepaid', 'gas bill', 'rent',
        'insurance premium', 'emi', 'loan', 'dth', 'maintenance',
        'society fee', 'property tax', 'wifi', 'utility',
        'lpg', 'cylinder', 'cable bill', 'landline', 'municipal',
        'building charge', 'annual subscription', 'hosting',
        'software license', 'cloud storage', 'security system',
        'generator', 'lift charge', 'solar', 'purifier service',
        'ac service', 'subscription bill'
    ],
    'Health': [
        'pharmacy', 'medicine', 'doctor', 'hospital', 'clinic',
        'blood test', 'lab test', 'checkup', 'dental', 'eye test',
        'gym', 'yoga', 'protein', 'supplement', 'vitamin',
        'calcium', 'physiotherapy', 'therapy', 'xray', 'mri',
        'scan', 'vaccine', 'ayurvedic', 'homeopathy', 'dietitian',
        'nutritionist', 'fitness', 'pulse oximeter', 'glucometer',
        'thermometer', 'first aid', 'sanitizer', 'optician',
        'glasses', 'contact lens', 'hearing aid', 'orthopedic',
        'dermatologist', 'skin treatment', 'hair treatment',
        'pediatric', 'gynecologist', 'psychiatrist', 'health'
    ],
    'Shopping': [
        'flipkart', 'myntra', 'ajio', 'meesho', 'snapdeal',
        'nykaa', 'purplle', 'clothing', 'clothes', 'fashion',
        'shoes', 'footwear', 'electronics', 'mobile phone',
        'laptop', 'furniture', 'home decor', 'appliance',
        'headphones', 'earbuds', 'smartwatch', 'tablet',
        'camera', 'printer', 'keyboard', 'monitor', 'hard disk',
        'pen drive', 'phone case', 'charger', 'power bank',
        'speaker', 'television', 'refrigerator', 'washing machine',
        'air conditioner', 'microwave', 'mixer', 'fan', 'lamp',
        'bedsheet', 'curtain', 'bathroom', 'stationery', 'shopping',
        'purchase online', 'order delivered', 'amazon shopping'
    ],
    'Income': [
        'salary', 'freelance', 'business income', 'dividend',
        'interest earned', 'rental income', 'bonus', 'refund',
        'cashback', 'investment return', 'stock profit',
        'mutual fund', 'fixed deposit', 'savings interest',
        'consulting fee received', 'commission earned',
        'affiliate', 'youtube earnings', 'blog income',
        'teaching income', 'tuition received', 'rent collected',
        'prize money', 'scholarship', 'insurance claim',
        'credit received', 'payment received', 'income'
    ],
    'Other': [
        'atm withdrawal', 'bank charge', 'transaction fee',
        'neft', 'rtgs', 'imps', 'wallet', 'cheque',
        'demand draft', 'account fee', 'kyc', 'pan card',
        'notary', 'stamp duty', 'legal fee', 'tax filing',
        'gst', 'income tax', 'customs', 'courier',
        'postal', 'printing', 'photocopy', 'passport fee',
        'visa fee', 'embassy', 'government fee'
    ]
}


# ============================================================
# STEP 1: RULE BASED CATEGORIZATION
# ============================================================

def rule_based_predict(description: str) -> dict:
    """
    Try to categorize using keyword matching first.
    
    HOW IT WORKS:
    We check if any keyword from our dictionary
    exists in the description text.
    
    If found → return that category with high confidence
    If not found → return None (ML will handle it)
    """
    
    text = description.lower().strip()
    
    # Track matches across categories
    matches = {}
    
    for category, keywords in CATEGORY_KEYWORDS.items():
        # Count how many keywords match
        match_count = sum(1 for kw in keywords if kw in text)
        if match_count > 0:
            matches[category] = match_count
    
    if not matches:
        # No keyword matched, ML will handle this
        return None
    
    # Pick category with most keyword matches
    best_category = max(matches, key=matches.get)
    
    # Confidence based on match count
    # More matches = higher confidence
    total_matches = sum(matches.values())
    confidence = round((matches[best_category] / total_matches) * 100, 2)
    
    # Minimum confidence of 85% for rule based
    confidence = max(confidence, 85.0)
    
    return {
        'category': best_category,
        'confidence': confidence,
        'method': 'rule_based'
    }


# ============================================================
# STEP 2: LOAD AND PREPARE ML TRAINING DATA
# ============================================================

def load_data():
    """Load training CSV file"""
    
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(
        current_dir, '..', 'data', 'training_data.csv'
    )
    
    df = pd.read_csv(data_path)
    
    print(f"Data loaded successfully!")
    print(f"Total examples: {len(df)}")
    print(f"Categories found: {df['category'].unique()}")
    print(f"Examples per category:")
    print(df['category'].value_counts())
    print()
    
    return df


def prepare_data(df):
    """Clean and split data for ML training"""
    
    # Clean text
    X = df['description'].str.lower().str.strip()
    X = X.str.replace(r'[^a-zA-Z\s]', '', regex=True)
    X = X.str.replace(r'\s+', ' ', regex=True)
    
    y = df['category']
    
    # Stratified split keeps proportions equal
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    print(f"Training examples: {len(X_train)}")
    print(f"Testing examples: {len(X_test)}")
    print()
    
    return X_train, X_test, y_train, y_test


# ============================================================
# STEP 3: TRAIN ML MODEL (backup for unknown descriptions)
# ============================================================

def train_model(X_train, y_train):
    """
    Train ML model as backup for descriptions
    that do not match any keyword rules.
    """
    
    model = Pipeline([
        ('tfidf', TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.9,
            sublinear_tf=True
        )),
        ('classifier', LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver='lbfgs',
            random_state=42
        ))
    ])
    
    print("Training the model...")
    model.fit(X_train, y_train)
    print("Model trained successfully!")
    print()
    
    return model


# ============================================================
# STEP 4: HYBRID PREDICTION (rules + ML combined)
# ============================================================

def hybrid_predict(description: str, model) -> dict:
    """
    Combine rule based and ML prediction.
    
    LOGIC:
    1. Try rule based first
    2. If rule based finds a match → use it
    3. If no match → use ML model
    
    This gives us:
    → High accuracy for known expense types (rules)
    → Flexibility for unknown expense types (ML)
    """
    
    # Try rule based first
    rule_result = rule_based_predict(description)
    
    if rule_result is not None:
        return rule_result
    
    # Fall back to ML
    cleaned = description.lower().strip()
    cleaned = ''.join(c for c in cleaned if c.isalpha() or c == ' ')
    
    category = model.predict([cleaned])[0]
    probabilities = model.predict_proba([cleaned])[0]
    confidence = round(max(probabilities) * 100, 2)
    
    return {
        'category': category,
        'confidence': confidence,
        'method': 'ml_model'
    }


# ============================================================
# STEP 5: EVALUATE MODEL
# ============================================================

def evaluate_model(model, X_test, y_test):
    """Test ML model accuracy on test data"""
    
    y_predicted = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_predicted)
    
    print(f"ML Model Accuracy (backup model): {accuracy * 100:.2f}%")
    print()
    print("Detailed Report:")
    print(classification_report(
        y_test,
        y_predicted,
        zero_division=0    # suppresses the UndefinedMetricWarning
    ))
    
    return accuracy


# ============================================================
# STEP 6: SAVE MODEL
# ============================================================

def save_model(model):
    """Save trained model and keyword rules to file"""
    
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, 'model.pkl')
    
    # Save both model AND keyword rules together
    # So predict.py can load everything at once
    save_object = {
        'ml_model': model,
        'keyword_rules': CATEGORY_KEYWORDS
    }
    
    with open(model_path, 'wb') as file:
        pickle.dump(save_object, file)
    
    print(f"Model saved successfully at: {model_path}")


# ============================================================
# STEP 7: TEST WITH CUSTOM EXAMPLES
# ============================================================

def test_with_examples(model):
    """Test hybrid system with our custom examples"""
    
    test_examples = [
        "mcdonalds burger",
        "uber cab booking",
        "netflix monthly subscription",
        "electricity bill payment",
        "apollo pharmacy medicine",
        "amazon online shopping",
        "salary credit this month",
        "random unknown payment xyz",
        "amazon prime video",
        "flipkart new order",
        "water bill payment",
        "gym membership fees"
    ]
    
    print("Testing with custom examples (Hybrid System):")
    print("━" * 55)
    
    for example in test_examples:
        result = hybrid_predict(example, model)
        method = result.get('method', 'unknown')
        
        print(f"Input:      '{example}'")
        print(f"Category:   {result['category']}")
        print(f"Confidence: {result['confidence']}%")
        print(f"Method:     {method}")
        print()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    
    print("=" * 50)
    print("SMART EXPENSE ANALYZER - MODEL TRAINING")
    print("=" * 50)
    print()
    
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    test_with_examples(model)
    save_model(model)
    
    print()
    print("=" * 50)
    print("ALL DONE! Hybrid model is ready to use.")
    print("=" * 50)