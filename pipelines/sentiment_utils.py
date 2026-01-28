"""
Sentiment Analysis Utilities
Wrapper for the trained DistilBERT sentiment model
"""
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import SENTIMENT_MODEL_DIR, ASPECT_KEYWORDS


class SentimentAnalyzer:
    """
    Sentiment analysis using trained DistilBERT model.
    
    Provides:
    - Overall sentiment score (0-1)
    - Aspect-based scores (food, service, ambiance, value)
    - Review keyword extraction
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
    
    def initialize(self):
        """Load the trained sentiment model."""
        print(" Initializing Sentiment Analyzer...")
        
        model_path = SENTIMENT_MODEL_DIR / "final_model"
        
        if not model_path.exists():
            print(f"    Model not found at {model_path}")
            print("   Using rule-based fallback for sentiment analysis")
            self.model_loaded = False
            return self
        
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
            
            print(f"   Loading model from: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.eval()
            self.model_loaded = True
            print("    Sentiment model loaded!")
        except Exception as e:
            print(f"    Error loading model: {e}")
            print("   Using rule-based fallback")
            self.model_loaded = False
        
        return self
    
    def analyze_text(self, text: str) -> float:
        """
        Analyze sentiment of a single text.
        
        Returns:
            Sentiment score 0-1 (0=negative, 1=positive)
        """
        if not text or len(text.strip()) < 10:
            return 0.5  # Neutral for short/empty text
        
        if self.model_loaded:
            return self._model_inference(text)
        else:
            return self._rule_based_sentiment(text)
    
    def _model_inference(self, text: str) -> float:
        """Use trained model for inference."""
        import torch
        
        # Tokenize
        inputs = self.tokenizer(
            text[:512],  # Limit length
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True
        )
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            
            # Assuming binary classification: idx 0=negative, idx 1=positive
            positive_prob = probs[0][1].item()
            return positive_prob
    
    def _rule_based_sentiment(self, text: str) -> float:
        """Fallback rule-based sentiment analysis."""
        text_lower = text.lower()
        
        positive_words = [
            "excellent", "amazing", "great", "good", "delicious", "tasty",
            "wonderful", "fantastic", "awesome", "love", "loved", "best",
            "recommend", "fresh", "nice", "friendly", "perfect", "superb"
        ]
        
        negative_words = [
            "bad", "terrible", "awful", "worst", "horrible", "poor",
            "disappointing", "bland", "stale", "rude", "slow", "dirty",
            "expensive", "overpriced", "cold", "mediocre", "average"
        ]
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total = positive_count + negative_count
        if total == 0:
            return 0.5  # Neutral
        
        return positive_count / total
    
    def analyze_reviews(self, reviews_text: str) -> Dict:
        """
        Analyze multiple reviews and extract insights.
        
        Args:
            reviews_text: Raw reviews string (can be list format)
            
        Returns:
            Dict with sentiment score, aspect scores, and keywords
        """
        # Parse reviews (handle different formats)
        reviews = self._parse_reviews(reviews_text)
        
        if not reviews:
            return self._empty_result()
        
        # Analyze each review
        sentiments = []
        aspect_mentions = {aspect: [] for aspect in ASPECT_KEYWORDS}
        keywords = []
        
        for review in reviews:
            # Overall sentiment
            sentiment = self.analyze_text(review)
            sentiments.append(sentiment)
            
            # Aspect extraction
            review_lower = review.lower()
            for aspect, keywords_list in ASPECT_KEYWORDS.items():
                for keyword in keywords_list:
                    if keyword in review_lower:
                        # Check if mention is positive or negative
                        aspect_mentions[aspect].append(sentiment)
                        break
            
            # Extract keywords
            keywords.extend(self._extract_keywords(review))
        
        # Calculate scores
        overall_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.5
        
        aspect_scores = {}
        for aspect, scores in aspect_mentions.items():
            if scores:
                aspect_scores[aspect] = sum(scores) / len(scores)
            else:
                aspect_scores[aspect] = 0.5  # Neutral if not mentioned
        
        # Get unique top keywords
        keyword_counts = {}
        for kw in keywords:
            keyword_counts[kw] = keyword_counts.get(kw, 0) + 1
        top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'sentiment_score': overall_sentiment,
            'review_count': len(reviews),
            'food_quality_score': aspect_scores.get('food_quality', 0.5),
            'service_score': aspect_scores.get('service', 0.5),
            'ambiance_score': aspect_scores.get('ambiance', 0.5),
            'value_score': aspect_scores.get('value_for_money', 0.5),
            'keywords': [kw for kw, _ in top_keywords],
            'positive_signals': [kw for kw, _ in top_keywords[:5] if keyword_counts[kw] >= 2],
            'negative_signals': []  # Would need more sophisticated analysis
        }
    
    def _parse_reviews(self, reviews_text: str) -> List[str]:
        """Parse reviews from various formats."""
        if not reviews_text:
            return []
        
        # Handle list format: ["review1", "review2"]
        if reviews_text.startswith('['):
            try:
                import ast
                reviews = ast.literal_eval(reviews_text)
                if isinstance(reviews, list):
                    # Extract text from nested format if needed
                    result = []
                    for r in reviews:
                        if isinstance(r, tuple) and len(r) >= 2:
                            result.append(str(r[1]))
                        elif isinstance(r, str):
                            result.append(r)
                    return result
            except:
                pass
        
        # Handle as single review or newline-separated
        if '\n' in reviews_text:
            return [r.strip() for r in reviews_text.split('\n') if r.strip()]
        
        return [reviews_text]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from review text."""
        text_lower = text.lower()
        
        all_keywords = []
        for aspect, keywords in ASPECT_KEYWORDS.items():
            all_keywords.extend(keywords)
        
        found = []
        for keyword in all_keywords:
            if keyword in text_lower:
                found.append(keyword)
        
        return found
    
    def _empty_result(self) -> Dict:
        """Return empty/neutral result."""
        return {
            'sentiment_score': 0.5,
            'review_count': 0,
            'food_quality_score': 0.5,
            'service_score': 0.5,
            'ambiance_score': 0.5,
            'value_score': 0.5,
            'keywords': [],
            'positive_signals': [],
            'negative_signals': []
        }


# Test function
def test_sentiment():
    """Test sentiment analyzer."""
    print("\n" + "=" * 60)
    print("TESTING SENTIMENT ANALYZER")
    print("=" * 60)
    
    analyzer = SentimentAnalyzer()
    analyzer.initialize()
    
    # Test texts
    tests = [
        "The food was absolutely delicious! Best biryani in town.",
        "Terrible experience. The service was rude and food was cold.",
        "It was okay, nothing special. Average food at average prices.",
        "Amazing ambiance, perfect for a romantic dinner. Loved the decor!"
    ]
    
    print("\n Single text analysis:")
    for text in tests:
        score = analyzer.analyze_text(text)
        sentiment = "positive" if score > 0.6 else "negative" if score < 0.4 else "neutral"
        print(f"   [{sentiment:8}] ({score:.2f}) {text[:50]}...")
    
    # Test multi-review analysis
    print("\n Multi-review analysis:")
    reviews = """
    Great food and amazing service! The butter chicken was perfect.
    The ambiance is cozy and romantic. Perfect for date night.
    A bit expensive but worth every penny. Will definitely come back.
    """
    result = analyzer.analyze_reviews(reviews)
    print(f"   Overall sentiment: {result['sentiment_score']:.2f}")
    print(f"   Food quality: {result['food_quality_score']:.2f}")
    print(f"   Service: {result['service_score']:.2f}")
    print(f"   Ambiance: {result['ambiance_score']:.2f}")
    print(f"   Keywords: {result['keywords']}")


if __name__ == "__main__":
    test_sentiment()
