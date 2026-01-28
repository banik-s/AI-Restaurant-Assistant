"""
Configuration settings for Multi-Agent Restaurant Recommendation System
"""
import os
from pathlib import Path

# =============================================================================
# PROJECT PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"
METADATA_DB_PATH = DATA_DIR / "restaurants_metadata.db"
MODELS_DIR = PROJECT_ROOT / "models"
SENTIMENT_MODEL_DIR = MODELS_DIR / "sentiment"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CHROMA_DB_DIR, SENTIMENT_MODEL_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# API KEYS
# =============================================================================
OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY",
    "sk-xxxxxxx"
    )

# =============================================================================
# MODEL SETTINGS
# =============================================================================
# Embedding Model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Sentiment Model
SENTIMENT_BASE_MODEL = "distilbert-base-uncased"
SENTIMENT_MODEL_NAME = "zomato-sentiment-distilbert"
SENTIMENT_MAX_LENGTH = 256
SENTIMENT_BATCH_SIZE = 32
SENTIMENT_EPOCHS = 3
SENTIMENT_LEARNING_RATE = 2e-5

# LLM Settings
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 1000

# =============================================================================
# DATA SETTINGS
# =============================================================================
RAW_CSV_FILENAME = "zomato2.csv"
RAW_CSV_PATH = RAW_DATA_DIR / RAW_CSV_FILENAME
PROCESSED_PARQUET_PATH = PROCESSED_DATA_DIR / "restaurants_processed.parquet"
SAMPLE_SIZE = 10000  # Use 10K sample for development

# ChromaDB Settings
CHROMA_COLLECTION_NAME = "zomato_restaurants"
CHROMA_TOP_K = 10

# =============================================================================
# COLUMN NAMES
# =============================================================================
COLUMNS = {
    "url": "url",
    "address": "address",
    "name": "name",
    "online_order": "online_order",
    "book_table": "book_table",
    "rate": "rate",
    "votes": "votes",
    "phone": "phone",
    "location": "location",
    "rest_type": "rest_type",
    "dish_liked": "dish_liked",
    "cuisines": "cuisines",
    "approx_cost": "approx_cost(for two people)",
    "reviews_list": "reviews_list",
    "menu_item": "menu_item",
    "listed_in_type": "listed_in(type)",
    "listed_in_city": "listed_in(city)",
}

# =============================================================================
# ASPECT KEYWORDS (for aspect-based sentiment analysis)
# =============================================================================
ASPECT_KEYWORDS = {
    "food_quality": [
        "food", "taste", "tasty", "delicious", "yummy", "bland", "fresh", 
        "stale", "spicy", "flavour", "flavor", "dish", "dishes", "meal",
        "authentic", "cuisine", "cooked", "prepared", "quality"
    ],
    "service": [
        "service", "staff", "waiter", "server", "attentive", "rude", 
        "polite", "courteous", "helpful", "friendly", "slow", "quick",
        "fast", "prompt", "waited", "waiting", "response"
    ],
    "ambiance": [
        "ambiance", "ambience", "atmosphere", "decor", "decoration",
        "interior", "cozy", "romantic", "loud", "noisy", "quiet", "music",
        "lighting", "clean", "dirty", "spacious", "crowded", "seating",
        "rooftop", "outdoor", "indoor", "view", "beautiful"
    ],
    "value_for_money": [
        "price", "expensive", "cheap", "affordable", "costly", "overpriced",
        "value", "worth", "money", "budget", "pocket", "reasonable"
    ],
    "wait_time": [
        "wait", "waiting", "time", "quick", "fast", "slow", "hour",
        "minutes", "delay", "delayed", "rushed"
    ]
}

# =============================================================================
# AMBIANCE KEYWORDS (for query understanding)
# =============================================================================
AMBIANCE_MAPPING = {
    "romantic": ["romantic", "cozy", "intimate", "candlelit", "date", "couple", "anniversary"],
    "family": ["family", "kids", "children", "child-friendly", "family-friendly"],
    "casual": ["casual", "chill", "relaxed", "informal", "hangout"],
    "fine_dining": ["fine dining", "upscale", "elegant", "fancy", "premium", "luxury"],
    "lively": ["lively", "energetic", "party", "fun", "music", "dance", "loud"],
    "quiet": ["quiet", "peaceful", "serene", "calm", "silent"],
    "outdoor": ["outdoor", "rooftop", "terrace", "garden", "open-air", "alfresco"],
}

# =============================================================================
# OCCASION KEYWORDS
# =============================================================================
OCCASION_MAPPING = {
    "date": ["date", "romantic", "anniversary", "valentine", "couple"],
    "family": ["family", "birthday", "celebration", "get-together", "reunion"],
    "business": ["business", "meeting", "client", "corporate", "professional"],
    "friends": ["friends", "hangout", "catching up", "reunion", "party"],
    "solo": ["solo", "alone", "myself", "quick bite"],
}

# =============================================================================
# INTENT TAXONOMY
# =============================================================================
INTENT_TYPES = [
    "DISCOVERY",           # General restaurant search
    "OCCASION_BASED",      # Specific occasion (anniversary, birthday)
    "COMPARISON",          # Compare restaurants
    "REFINEMENT",          # Modify previous search
    "CUISINE_EXPLORATION", # Explore specific cuisine
    "BUDGET_OPTIMIZATION", # Find budget-friendly options
]

# =============================================================================
# PROMPT TEMPLATES
# =============================================================================
INTENT_CLASSIFICATION_PROMPT = """You are an intent classifier for a restaurant recommendation system.

Given a user query, classify it into ONE of these intents:
- DISCOVERY: General search for restaurants without specific occasion
- OCCASION_BASED: Looking for restaurants for a specific event (date, anniversary, birthday, meeting)
- COMPARISON: Comparing specific restaurants or asking "which is better"
- REFINEMENT: Modifying a previous search (closer, cheaper, vegetarian version)
- CUISINE_EXPLORATION: Specifically exploring a cuisine type
- BUDGET_OPTIMIZATION: Finding cheaper/budget-friendly options

Also extract:
1. Confidence score (0-100)
2. Implicit preferences if any

User Query: "{query}"

Respond in this exact JSON format:
{{
    "intent": "<INTENT_TYPE>",
    "confidence": <0-100>,
    "implicit_preferences": ["preference1", "preference2"]
}}
"""

PRIORITY_INFERENCE_PROMPT = """Based on the user's intent and query, infer priority rankings for restaurant attributes.

Intent: {intent}
Query: "{query}"

Assign priority levels (HIGH, MEDIUM, LOW) to each attribute:
- ambiance: How important is the atmosphere/setting?
- food_quality: How important is the food taste/quality?
- service: How important is the service quality?
- price: How sensitive is the user to price? (HIGH = very price-conscious)
- location: How important is the specific location?

Respond in this exact JSON format:
{{
    "ambiance": "<HIGH|MEDIUM|LOW>",
    "food_quality": "<HIGH|MEDIUM|LOW>",
    "service": "<HIGH|MEDIUM|LOW>",
    "price": "<HIGH|MEDIUM|LOW>",
    "location": "<HIGH|MEDIUM|LOW>"
}}
"""

EXPLANATION_GENERATION_PROMPT = """Generate a natural, conversational recommendation explanation.

Restaurant Data:
- Name: {name}
- Cuisines: {cuisines}
- Location: {location}
- Rating: {rating}/5 ({votes} reviews)
- Cost for Two: ₹{cost}
- Confidence Score: {confidence}%

Matching Signals:
{signals}

User's Intent: {intent}
User's Priorities: {priorities}

Generate a 2-3 sentence explanation of why this restaurant is recommended. 
Include specific evidence from signals. Be enthusiastic but honest.
If there are concerns, mention them briefly at the end.
"""
