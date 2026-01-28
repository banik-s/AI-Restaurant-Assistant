import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import OPENAI_API_KEY, LLM_MODEL, EXPLANATION_GENERATION_PROMPT
from models.schemas import (
    Restaurant, ScoredRestaurant, RankedRestaurant, 
    Priorities, PriorityLevel, RankingResult, IntentType
)
from pipelines.sentiment_utils import SentimentAnalyzer


class RankingEngine:
    """
    Agent 4: Intelligent ranking and recommendation engine.
    
    Responsibilities:
    - Run sentiment analysis on fetched restaurants
    - Calculate scores based on user intent and priorities
    - Rank restaurants with confidence scores
    - Generate explained recommendations using LLM
    """
    
    def __init__(self):
        self.sentiment_analyzer = None
        self.llm_client = None
    
    def initialize(self):
        """Initialize sentiment analyzer and LLM client."""
        print(" Initializing Ranking Engine...")
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = SentimentAnalyzer()
        self.sentiment_analyzer.initialize()
        
        # Initialize LLM client
        try:
            from openai import OpenAI
            self.llm_client = OpenAI(api_key=OPENAI_API_KEY)
            print("    LLM client ready")
        except Exception as e:
            print(f"    LLM client error: {e}")
            self.llm_client = None
        
        print(" Ranking Engine ready!")
        return self
    
    def analyze_restaurants(self, restaurants: List[Restaurant]) -> Dict[int, ScoredRestaurant]:
        
        scored = {}
        
        for restaurant in restaurants:
            if restaurant.reviews_list:
                analysis = self.sentiment_analyzer.analyze_reviews(restaurant.reviews_list)
            else:
                analysis = {
                    'sentiment_score': restaurant.rating / 5.0,
                    'food_quality_score': restaurant.rating / 5.0,
                    'service_score': 0.5,
                    'ambiance_score': 0.5,
                    'value_score': 0.5,
                    'keywords': [],
                    'positive_signals': [],
                    'negative_signals': []
                }
            
            scored[restaurant.id] = ScoredRestaurant(
                restaurant=restaurant,
                sentiment_score=analysis['sentiment_score'],
                food_quality_score=analysis['food_quality_score'],
                service_score=analysis['service_score'],
                ambiance_score=analysis['ambiance_score'],
                value_score=analysis['value_score'],
                review_keywords=analysis['keywords'],
                positive_signals=analysis['positive_signals'],
                negative_signals=analysis['negative_signals']
            )
        
        return scored
    
    def calculate_score(self, scored_restaurant: ScoredRestaurant, intent: IntentType, priorities: Priorities) -> Tuple[float, str]:
        
        r = scored_restaurant.restaurant
        s = scored_restaurant
        
        # Base score from rating and sentiment
        base_score = (
            (r.rating / 5.0) * 0.30 +
            s.sentiment_score * 0.25 +
            min(r.votes / 1000, 1.0) * 0.10  # Normalized votes
        )
        
        boost = 0.0
        penalty = 0.0
        reasons = []
        
        # Intent-based boost
        if intent == IntentType.NEW_SEARCH:
            # General discovery - balanced approach
            boost += s.food_quality_score * 0.15
            boost += s.service_score * 0.10
        
        elif intent in [IntentType.QUESTION_ABOUT_RESULTS, IntentType.COMPARISON]:
            # Focus on detail/comparison - use all signals
            boost += s.food_quality_score * 0.10
            boost += s.service_score * 0.10
            boost += s.ambiance_score * 0.10
        
        # Priority-based adjustments
        priority_weight = {
            PriorityLevel.HIGH: 0.30,
            PriorityLevel.MEDIUM: 0.15,
            PriorityLevel.LOW: 0.05
        }
        
        # Ambiance priority
        weight = priority_weight[priorities.ambiance]
        boost += s.ambiance_score * weight
        if priorities.ambiance == PriorityLevel.HIGH:
            if s.ambiance_score > 0.7:
                reasons.append("High ambiance score matches your preference")
        
        # Food quality priority  
        weight = priority_weight[priorities.food_quality]
        boost += s.food_quality_score * weight
        if priorities.food_quality == PriorityLevel.HIGH:
            if s.food_quality_score > 0.7:
                reasons.append("Excellent food quality")
        
        # Service priority
        weight = priority_weight[priorities.service]
        boost += s.service_score * weight
        
        # Price sensitivity
        if priorities.price == PriorityLevel.HIGH:
            if r.approx_cost > 1000:
                penalty += 0.15
                if "expensive" in s.negative_signals:
                    penalty += 0.10
            elif r.approx_cost < 500:
                boost += 0.10
                reasons.append("Budget-friendly option")
        
        # Location priority already handled by SQL filtering
        
        final_score = base_score + boost - penalty
        final_score = max(0.1, min(1.0, final_score))  # Clamp to 0.1-1.0
        
        reason = ". ".join(reasons) if reasons else "Good overall match"
        
        return final_score, reason
    
    def rank(self, restaurants: List[Restaurant], scored: Dict[int, ScoredRestaurant], intent: IntentType, priorities: Priorities, top_k: int = 5) -> List[RankedRestaurant]:
        
        results = []
        
        for restaurant in restaurants:
            scored_r = scored.get(restaurant.id)
            if not scored_r:
                # Create default scored restaurant
                scored_r = ScoredRestaurant(restaurant=restaurant)
            
            final_score, reason = self.calculate_score(scored_r, intent, priorities)
            
            results.append({
                'restaurant': restaurant,
                'scored': scored_r,
                'final_score': final_score,
                'reason': reason
            })
        
        # Sort by score
        results.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Convert to RankedRestaurant
        ranked = []
        for i, item in enumerate(results[:top_k]):
            ranked.append(RankedRestaurant(
                restaurant=item['restaurant'],
                rank=i + 1,
                confidence=int(item['final_score'] * 100),
                final_score=item['final_score'],
                explanation="",  # Will be filled by generate_explanations
                scores={
                    'sentiment': item['scored'].sentiment_score,
                    'food_quality': item['scored'].food_quality_score,
                    'service': item['scored'].service_score,
                    'ambiance': item['scored'].ambiance_score,
                    'value': item['scored'].value_score
                },
                popular_dishes=item['restaurant'].dish_liked.split(',')[:5] if item['restaurant'].dish_liked else [],
                concerns=item['scored'].negative_signals[:3],
                why_recommended=item['reason']
            ))
        
        return ranked
    
    def generate_explanation(self, ranked: RankedRestaurant, intent: IntentType, priorities: Priorities) -> str:
        if not self.llm_client:
            return self.fallback_explanation(ranked)
        
        r = ranked.restaurant
        
        # Build signals string
        signals = []
        if ranked.scores.get('sentiment', 0) > 0.7:
            signals.append(f"Positive reviews ({int(ranked.scores['sentiment']*100)}%)")
        if ranked.scores.get('food_quality', 0) > 0.7:
            signals.append(f"High food quality score ({int(ranked.scores['food_quality']*100)}%)")
        if ranked.scores.get('ambiance', 0) > 0.7:
            signals.append(f"Great ambiance ({int(ranked.scores['ambiance']*100)}%)")
        if ranked.popular_dishes:
            signals.append(f"Popular dishes: {', '.join(ranked.popular_dishes[:3])}")
        if r.online_order:
            signals.append("Online ordering available")
        if r.book_table:
            signals.append("Table booking available")
        
        if ranked.concerns:
            signals.append(f"Note: Some reviews mention {', '.join(ranked.concerns[:2])}")
        
        prompt = EXPLANATION_GENERATION_PROMPT.format(
            name=r.name,
            cuisines=r.cuisines,
            location=r.location,
            rating=r.rating,
            votes=r.votes,
            cost=r.approx_cost,
            confidence=ranked.confidence,
            signals="\n".join(f"- {s}" for s in signals),
            intent=intent.value,
            priorities=priorities.to_dict()
        )
        
        try:
            response = self.llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f" LLM error: {e}")
            return self.fallback_explanation(ranked)
    
    def fallback_explanation(self, ranked: RankedRestaurant) -> str:
        r = ranked.restaurant
        
        parts = [
            f"{r.name} in {r.location} offers {r.cuisines}.",
            f"With a rating of {r.rating}/5 from {r.votes} reviews, "
            f"it's priced at ₹{r.approx_cost} for two."
        ]
        
        if ranked.scores.get('sentiment', 0) > 0.7:
            parts.append(f"Reviews are mostly positive ({int(ranked.scores['sentiment']*100)}% satisfaction).")
        
        if ranked.popular_dishes:
            parts.append(f"Popular dishes include {', '.join(ranked.popular_dishes[:3])}.")
        
        if ranked.concerns:
            parts.append(f"Note: Some mention {ranked.concerns[0]}.")
        
        return " ".join(parts)
    
    def generate_explanations(self, ranked_list: List[RankedRestaurant], intent: IntentType, priorities: Priorities, generate_llm: bool = True) -> List[RankedRestaurant]:
        for i, ranked in enumerate(ranked_list):
            if generate_llm and i < 3 and self.llm_client:
                ranked.explanation = self.generate_explanation(ranked, intent, priorities)
            else:
                ranked.explanation = self.fallback_explanation(ranked)
        
        return ranked_list
    
    def generate_comparison(self, restaurant_a: RankedRestaurant, restaurant_b: RankedRestaurant) -> str:
        """Generate side-by-side comparison of two restaurants."""
        a, b = restaurant_a.restaurant, restaurant_b.restaurant
        sa, sb = restaurant_a.scores, restaurant_b.scores
        
        lines = [
            f"## Comparison: {a.name} vs {b.name}\n",
            "| Aspect | " + a.name[:15] + " | " + b.name[:15] + " |",
            "|--------|" + "-" * 15 + "|" + "-" * 15 + "|",
            f"| Rating | {a.rating}/5 | {b.rating}/5 |",
            f"| Cost | ₹{a.approx_cost} | ₹{b.approx_cost} |",
            f"| Location | {a.location} | {b.location} |",
            f"| Food Score | {int(sa.get('food_quality', 0.5)*100)}% | {int(sb.get('food_quality', 0.5)*100)}% |",
            f"| Service | {int(sa.get('service', 0.5)*100)}% | {int(sb.get('service', 0.5)*100)}% |",
            f"| Ambiance | {int(sa.get('ambiance', 0.5)*100)}% | {int(sb.get('ambiance', 0.5)*100)}% |",
        ]
        
        # Recommendation
        better = a if a.rating > b.rating else b
        lines.append(f"\n**Overall**: {better.name} has a higher rating.")
        
        # Best for
        if sa.get('ambiance', 0) > sb.get('ambiance', 0) + 0.1:
            lines.append(f"\n**For ambiance**: {a.name} scores better.")
        elif sb.get('ambiance', 0) > sa.get('ambiance', 0) + 0.1:
            lines.append(f"\n**For ambiance**: {b.name} scores better.")
        
        if a.approx_cost < b.approx_cost:
            lines.append(f"\n**Budget pick**: {a.name} is more affordable.")
        else:
            lines.append(f"\n**Budget pick**: {b.name} is more affordable.")
        
        return "\n".join(lines)
    
    def process(self, restaurants: List[Restaurant], intent: IntentType, priorities: Priorities, top_k: int = 5, generate_llm_explanations: bool = True) -> RankingResult:
        start_time = time.time()
        
        if not restaurants:
            return RankingResult(
                recommendations=[],
                reasoning="No restaurants to rank",
                processing_time_ms=0
            )
        
        # Step 1: Analyze sentiment
        print(f"   Analyzing {len(restaurants)} restaurants...")
        scored = self.analyze_restaurants(restaurants)
        
        # Step 2: Rank
        print("   Ranking by intent and priorities...")
        ranked = self.rank(restaurants, scored, intent, priorities, top_k)
        
        # Step 3: Generate explanations
        if generate_llm_explanations:
            print("   Generating explanations...")
        ranked = self.generate_explanations(ranked, intent, priorities, generate_llm_explanations)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Build reasoning
        reasoning = f"Ranked {len(restaurants)} restaurants based on {intent.value} intent. "
        high_priorities = [k for k, v in priorities.to_dict().items() if v == "HIGH"]
        if high_priorities:
            reasoning += f"Prioritized: {', '.join(high_priorities)}."
        
        return RankingResult(
            recommendations=ranked,
            reasoning=reasoning,
            processing_time_ms=elapsed_ms
        )


# Import Tuple for type hints
from typing import Tuple


# Test function  
def test_ranking_engine():
    
    engine = RankingEngine()
    engine.initialize()
    
    # Create test restaurants
    test_restaurants = [
        Restaurant(id=1, name="Tandoor Palace", location="Koramangala",
                   cuisines="North Indian, Mughlai", approx_cost=900, rating=4.5, votes=200,
                   dish_liked="Butter Chicken, Biryani, Naan"),
        Restaurant(id=2, name="Pizza Corner", location="Koramangala",
                   cuisines="Italian, Pizza", approx_cost=600, rating=4.0, votes=150,
                   dish_liked="Margherita, Pepperoni"),
        Restaurant(id=3, name="Thai Orchid", location="Koramangala",
                   cuisines="Thai, Asian", approx_cost=1200, rating=4.3, votes=180,
                   dish_liked="Pad Thai, Tom Yum"),
    ]
    
    # Test ranking
    print("\n Testing ranking pipeline")
    result = engine.process(
        restaurants=test_restaurants,
        intent=IntentType.NEW_SEARCH,
        priorities=Priorities(
            ambiance=PriorityLevel.HIGH,
            food_quality=PriorityLevel.HIGH,
            price=PriorityLevel.MEDIUM
        ),
        top_k=3,
        generate_llm_explanations=False  # Skip LLM for test
    )
    
    print(f"\n   Processing time: {result.processing_time_ms:.0f}ms")
    print(f"   Reasoning: {result.reasoning}")
    
    for rec in result.recommendations:
        print(f"\n   #{rec.rank} {rec.restaurant.name} (Confidence: {rec.confidence}%)")
        print(f"      Scores: {rec.scores}")
        print(f"      Explanation: {rec.explanation[:100]}...")


if __name__ == "__main__":
    test_ranking_engine()
