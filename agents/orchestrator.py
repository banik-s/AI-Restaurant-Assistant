import sys
import re
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import OPENAI_API_KEY, LLM_MODEL, INTENT_CLASSIFICATION_PROMPT, PRIORITY_INFERENCE_PROMPT
from models.schemas import (
    IntentType, AgentRoute, PriorityLevel,
    Constraints, Priorities, IntentResult,
    Restaurant, RankedRestaurant, ConversationTurn
)
from models.session_state import SessionState
from agents.sql_builder import SQLQueryBuilder
from agents.session_rag import SessionRAG
from agents.ranking_engine import RankingEngine


class ConversationalOrchestrator:
    """
    Agent 1: Master conversational orchestrator.
    
    Responsibilities:
    - Classify user intent (NEW_SEARCH, REFINE_SEARCH, QUESTION, COMPARISON)
    - Extract constraints from natural language
    - Route to appropriate agents
    - Manage session state
    - Format responses
    """
    
    def __init__(self):
        self.session: Optional[SessionState] = None
        self.sql_builder: Optional[SQLQueryBuilder] = None
        self.session_rag: Optional[SessionRAG] = None
        self.ranking_engine: Optional[RankingEngine] = None
        self.llm_client = None
        self._wants_other_options = False  # Track when user wants different restaurants
    
    def initialize(self, session_id: str = None) -> 'ConversationalOrchestrator':
        print(" Initializing Conversational Orchestrator...")
        
        # Initialize sub-agents
        self.sql_builder = SQLQueryBuilder()
        self.sql_builder.initialize()
        
        self.session_rag = SessionRAG()
        self.session_rag.initialize()
        
        self.ranking_engine = RankingEngine()
        self.ranking_engine.initialize()
        
        # Initialize LLM client for intent classification
        try:
            from openai import OpenAI
            self.llm_client = OpenAI(api_key=OPENAI_API_KEY)
        except Exception as e:
            print(f" LLM client error: {e}")
        
        # Initialize session
        self.session = SessionState(session_id=session_id) if session_id else SessionState()
        
        print(f" Orchestrator ready! Session: {self.session.session_id[:8]}")
        return self
    
    def classify_intent(self, query: str) -> IntentResult:
        
        query_lower = query.lower()
        
        intent, route = self.quick_classify(query_lower)
        
        constraints = self.extract_constraints(query)
        
        if intent == IntentType.NEW_SEARCH:
            if self.session.needs_new_fetch(constraints):
                route = AgentRoute.FULL_PIPELINE
            elif self.session.can_filter_in_memory(constraints):
                intent = IntentType.REFINE_SEARCH
                route = AgentRoute.CACHED_FILTER
        
        # Infer priorities
        priorities = self.infer_priorities(query, intent)
        
        # Extract references (for QUESTION/COMPARISON)
        reference = self.extract_reference(query_lower)
        
        return IntentResult(
            intent=intent,
            confidence=85,
            route=route,
            constraints=constraints,
            priorities=priorities,
            reference=reference
        )
    
    def quick_classify(self, query_lower: str) -> tuple:        
        # COMPARISON patterns
        comparison_patterns = [
            r'compare\s+', r'vs\s+', r'versus\s+',
            r'which\s+(one\s+)?is\s+better', r'between\s+'
        ]
        if any(re.search(p, query_lower) for p in comparison_patterns):
            return IntentType.COMPARISON, AgentRoute.CACHED_RETRIEVE
        
        # QUESTION patterns (about previous results)
        question_patterns = [
            r'tell\s+me\s+(more\s+)?about', r'what\s+about\s+the',
            r'details\s+(about|of)', r'more\s+info',
            r'first\s+one', r'second\s+one', r'third\s+one',
            r'phone', r'address', r'menu', r'location\s+of'
        ]
        if self.session and self.session.has_cache():
            if any(re.search(p, query_lower) for p in question_patterns):
                return IntentType.QUESTION_ABOUT_RESULTS, AgentRoute.CACHED_RETRIEVE
        
        # OTHER OPTIONS patterns - user wants NEW different restaurants
        # This should trigger NEW_SEARCH, not cache filter!
        other_options_patterns = [
            r'other\s+', r'another\s+', r'different\s+',
            r'more\s+options', r'more\s+suggestions', r'more\s+recommendations',
            r'something\s+else', r'anything\s+else', r'alternatives',
            r'give\s+me\s+more', r'show\s+more', r'any\s+other'
        ]
        if any(re.search(p, query_lower) for p in other_options_patterns):
            self._wants_other_options = True
            return IntentType.NEW_SEARCH, AgentRoute.FULL_PIPELINE
        
        # REFINEMENT patterns (modify current search)
        refinement_patterns = [
            r'show\s+me\s+(only\s+)?', r'filter\s+',
            r'cheaper', r'under\s+[₹Rs.]*\d+', r'below\s+[₹Rs.]*\d+',
            r'vegetarian', r'vegan', r'with\s+',
            r'from\s+these', r'among\s+these',
            r'non[- ]?veg', r'table\s+booking'
        ]
        if self.session and self.session.has_cache():
            if any(re.search(p, query_lower) for p in refinement_patterns):
                return IntentType.REFINE_SEARCH, AgentRoute.CACHED_FILTER
        
        if len(query_lower.split()) < 3 and not any(
            w in query_lower for w in ['restaurant', 'food', 'eat', 'dinner', 'lunch']
        ):
            return IntentType.CLARIFICATION_NEEDED, AgentRoute.CLARIFY
        
        # Default: NEW_SEARCH
        self._wants_other_options = False
        return IntentType.NEW_SEARCH, AgentRoute.FULL_PIPELINE
    
    def extract_constraints(self, query: str) -> Constraints:
        query_lower = query.lower()
        constraints = Constraints()
        
        known_locations = self.sql_builder.get_known_locations() if self.sql_builder else []
        for loc in known_locations:
            if loc.lower() in query_lower:
                constraints.location = loc
                break
        
        if not constraints.location and self.sql_builder:
            # Common location mentions
            location_patterns = [
                r'in\s+(\w+)', r'at\s+(\w+)', r'around\s+(\w+)',
                r'near\s+(\w+)', r'(\w+)\s+area'
            ]
            for pattern in location_patterns:
                match = re.search(pattern, query_lower)
                if match:
                    loc = self.sql_builder.normalize_location(match.group(1))
                    if loc:
                        constraints.location = loc
                        break
        
        # Cuisine extraction
        known_cuisines = self.sql_builder.get_known_cuisines() if self.sql_builder else []
        found_cuisines = []
        for cuisine in known_cuisines:
            if cuisine.lower() in query_lower:
                found_cuisines.append(cuisine)
        if found_cuisines:
            constraints.cuisines = found_cuisines
        
        # Dish extraction (biryani, pizza, butter chicken, etc.)
        common_dishes = [
            'biryani', 'pizza', 'burger', 'dosa', 'idli', 'pasta', 'noodles',
            'momos', 'thali', 'kebab', 'tandoori', 'butter chicken', 'paneer',
            'naan', 'paratha', 'fried rice', 'manchurian', 'chow mein',
            'shawarma', 'sandwich', 'salad', 'soup', 'ice cream', 'cake',
            'coffee', 'tea', 'juice', 'smoothie', 'chicken', 'mutton', 'fish',
            'prawns', 'crab', 'lobster', 'steak', 'sushi', 'ramen', 'pho',
            'dim sum', 'spring roll', 'samosa', 'chaat', 'pav bhaji', 'vada pav',
            'pani puri', 'bhel', 'chole', 'rajma', 'dal', 'korma', 'tikka',
            'malai', 'seekh', 'rolls', 'wrap', 'fries', 'wings', 'ribs'
        ]
        found_dishes = []
        for dish in common_dishes:
            if dish in query_lower:
                found_dishes.append(dish)
        if found_dishes:
            constraints.dishes = found_dishes
        
        # Budget extraction
        budget_patterns = [
            r'under\s+[₹Rs.]*\s*(\d+)', r'below\s+[₹Rs.]*\s*(\d+)',
            r'less\s+than\s+[₹Rs.]*\s*(\d+)', r'[₹Rs.]*\s*(\d+)\s+budget',
            r'within\s+[₹Rs.]*\s*(\d+)', r'max\s+[₹Rs.]*\s*(\d+)',
            r'upto\s+[₹Rs.]*\s*(\d+)', r'[₹Rs.]\s*(\d+)'
        ]
        for pattern in budget_patterns:
            match = re.search(pattern, query_lower)
            if match:
                constraints.budget_max = int(match.group(1))
                break
        
        # Rating extraction
        rating_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:\+|plus)?\s*(?:star|rating)',
            r'rating\s*(?:above|over|at least)\s*(\d+(?:\.\d+)?)',
            r'above\s+(\d+(?:\.\d+)?)\s+star'
        ]
        for pattern in rating_patterns:
            match = re.search(pattern, query_lower)
            if match:
                constraints.min_rating = float(match.group(1))
                break
        
        # Boolean features
        if 'online order' in query_lower or 'delivery' in query_lower:
            constraints.online_order = True
        
        if 'book table' in query_lower or 'reservation' in query_lower or 'table booking' in query_lower:
            constraints.book_table = True
        
        # Dietary extraction  
        dietary = []
        if 'vegetarian' in query_lower or 'veg ' in query_lower:
            dietary.append('vegetarian')
        if 'vegan' in query_lower:
            dietary.append('vegan')
        if 'non-veg' in query_lower or 'non veg' in query_lower:
            dietary.append('non-veg')
        if dietary:
            constraints.dietary = dietary
        
        # Ambiance extraction
        ambiance = []
        ambiance_words = ['romantic', 'family', 'casual', 'fancy', 'rooftop', 'outdoor', 'quiet', 'lively']
        for word in ambiance_words:
            if word in query_lower:
                ambiance.append(word)
        if ambiance:
            constraints.ambiance = ambiance
        
        # Occasion extraction
        occasion_words = ['anniversary', 'birthday', 'date', 'meeting', 'friends']
        for word in occasion_words:
            if word in query_lower:
                constraints.occasion = word
                break
        
        # Log extracted constraints for debugging
        print(f"   Extracted constraints: {constraints.to_dict()}")
        
        return constraints
    
    def infer_priorities(self, query: str, intent: IntentType) -> Priorities:
        """Infer user priorities from query and intent."""
        query_lower = query.lower()
        priorities = Priorities()
        
        # Ambiance priority
        if any(w in query_lower for w in ['romantic', 'ambiance', 'atmosphere', 'anniversary', 'date', 'cozy']):
            priorities.ambiance = PriorityLevel.HIGH
        
        # Food quality priority
        if any(w in query_lower for w in ['best', 'authentic', 'delicious', 'tasty', 'quality', 'craving']):
            priorities.food_quality = PriorityLevel.HIGH
        
        # Service priority
        if any(w in query_lower for w in ['service', 'quick', 'fast', 'friendly']):
            priorities.service = PriorityLevel.HIGH
        
        # Price priority
        if any(w in query_lower for w in ['cheap', 'budget', 'affordable', 'under', 'below']):
            priorities.price = PriorityLevel.HIGH
        elif any(w in query_lower for w in ['fine dining', 'premium', 'expensive']):
            priorities.price = PriorityLevel.LOW
        
        # Location priority
        if any(w in query_lower for w in ['near', 'close', 'walking distance']):
            priorities.location = PriorityLevel.HIGH
        
        return priorities
    
    def extract_reference(self, query_lower: str) -> Optional[str]:
        """Extract restaurant reference from query."""
        # Positional references
        position_map = {
            'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
            '1st': 1, '2nd': 2, '3rd': 3, '4th': 4, '5th': 5
        }
        
        for word, pos in position_map.items():
            if word in query_lower:
                result = self.session.get_restaurant_by_position(pos) if self.session else None
                if result:
                    return str(result.restaurant.id)
        
        # Try to find restaurant name
        if self.session and self.session.last_recommendations:
            for rec in self.session.last_recommendations:
                if rec.restaurant.name.lower() in query_lower:
                    return str(rec.restaurant.id)
        
        return None
    
    def route_to_agents(self, intent_result: IntentResult) -> Dict:
        """Route query to appropriate agents and get results."""
        route = intent_result.route
        constraints = intent_result.constraints
        priorities = intent_result.priorities
        intent = intent_result.intent
        
        start_time = time.time()
        result = {}
        
        if route == AgentRoute.FULL_PIPELINE:
            # Full pipeline: Agent 2 → Agent 3 → Agent 4
            result = self.full_pipeline(constraints, priorities, intent)
            
        elif route == AgentRoute.CACHED_FILTER:
            # Use cache: Agent 3 → Agent 4
            result = self.cached_filter_pipeline(constraints, priorities, intent)
            
        elif route == AgentRoute.CACHED_RETRIEVE:
            # Retrieve specific: Agent 3 → Agent 4
            result = self.cached_retrieve_pipeline(intent_result)
            
        elif route == AgentRoute.CLARIFY:
            result = {'needs_clarification': True}
        
        result['timing_ms'] = (time.time() - start_time) * 1000
        result['route'] = route.value
        
        return result
    
    def full_pipeline(self, constraints: Constraints, priorities: Priorities, intent: IntentType) -> Dict:
        """Full pipeline: Database fetch -> Index -> Rank."""
        print(f" Full pipeline: NEW_SEARCH")
        
        # Step 1: Agent 2 - SQL Query
        sql_result = self.sql_builder.execute_with_fallback(constraints)
        print(f"   Agent 2: Fetched {sql_result.count} restaurants in {sql_result.execution_time_ms:.0f}ms")
        
        # If user wants "other" options, exclude already shown restaurants
        if self._wants_other_options and self.session.last_recommendations:
            shown_ids = {rec.restaurant.id for rec in self.session.last_recommendations}
            original_count = len(sql_result.restaurants)
            sql_result.restaurants = [r for r in sql_result.restaurants if r.id not in shown_ids]
            print(f"   Excluded {original_count - len(sql_result.restaurants)} already-shown restaurants")
            self._wants_other_options = False  # Reset flag
        
        if sql_result.count == 0 or len(sql_result.restaurants) == 0:
            location = constraints.location or "this area"
            dishes = ', '.join(constraints.dishes) if constraints.dishes else "this type"
            return {
                'recommendations': [],
                'message': f'No other {dishes} restaurants found in {location}. You have seen all available options. Try searching in a different area like Indiranagar, BTM, or HSR.',
                'fallback_used': False
            }
        
        # Update session cache
        self.session.update_cache(sql_result.restaurants)
        self.session.current_constraints = constraints
        
        # Step 2: Agent 3 - Index restaurants
        index_time = self.session_rag.index_restaurants(
            sql_result.restaurants, 
            self.session.session_id
        )
        print(f"   Agent 3: Indexed in {index_time:.0f}ms")
        
        # Step 3: Agent 4 - Rank and explain
        ranking_result = self.ranking_engine.process(
            restaurants=sql_result.restaurants,
            intent=intent,
            priorities=priorities,
            top_k=5,
            generate_llm_explanations=True
        )
        print(f"   Agent 4: Ranked in {ranking_result.processing_time_ms:.0f}ms")
        
        # Update session
        self.session.last_recommendations = ranking_result.recommendations
        
        return {
            'recommendations': ranking_result.recommendations,
            'count': sql_result.count,
            'fallback_used': sql_result.fallback_used,
            'fallback_reason': sql_result.fallback_reason,
            'reasoning': ranking_result.reasoning
        }
    
    def cached_filter_pipeline(self, constraints: Constraints, priorities: Priorities, intent: IntentType) -> Dict:
        print(f" Cached pipeline: REFINE_SEARCH (using {len(self.session.cached_restaurants)} cached)")
        
        # Step 1: Filter cached restaurants (Agent 3)
        rag_result = self.session_rag.filter_cached(
            session_id=self.session.session_id,
            budget_max=constraints.budget_max,
            budget_min=constraints.budget_min,
            min_rating=constraints.min_rating,
            cuisines=constraints.cuisines,
            dietary=constraints.dietary,
            online_order=constraints.online_order,
            book_table=constraints.book_table
        )
        print(f"   Agent 3: Filtered to {len(rag_result.restaurants)} restaurants")
        
        if not rag_result.restaurants:
            return {
                'recommendations': [],
                'message': 'No restaurants match those filters. Try relaxing some constraints.',
                'from_cache': True
            }
        
        # Step 2: Re-rank filtered results (Agent 4)
        # Use lightweight ranking (no LLM calls)
        ranking_result = self.ranking_engine.process(
            restaurants=rag_result.restaurants,
            intent=intent,
            priorities=priorities,
            top_k=5,
            generate_llm_explanations=False  # Fast mode
        )
        print(f"   Agent 4: Ranked in {ranking_result.processing_time_ms:.0f}ms")
        
        # Update session
        self.session.current_constraints = self.session.current_constraints.merge_with(constraints)
        self.session.last_recommendations = ranking_result.recommendations
        
        return {
            'recommendations': ranking_result.recommendations,
            'count': len(rag_result.restaurants),
            'from_cache': True,
            'filter_applied': rag_result.filter_applied,
            'reasoning': ranking_result.reasoning
        }
    
    def cached_retrieve_pipeline(self, intent_result: IntentResult) -> Dict:
        """Cached retrieve pipeline: Get specific restaurant(s)."""
        intent = intent_result.intent
        reference = intent_result.reference
        
        if intent == IntentType.COMPARISON:
            return self.handle_comparison(intent_result)
        
        # Get referenced restaurant
        if reference:
            restaurant = self.session_rag.get_by_id(
                self.session.session_id, 
                int(reference)
            )
            if restaurant:
                # Find in last recommendations for full details
                for rec in self.session.last_recommendations:
                    if rec.restaurant.id == int(reference):
                        return {
                            'recommendations': [rec],
                            'is_detail': True
                        }
                
                # Create basic recommendation
                return {
                    'recommendations': [RankedRestaurant(
                        restaurant=restaurant,
                        rank=1,
                        confidence=80,
                        final_score=0.8,
                        explanation="",
                        scores={}
                    )],
                    'is_detail': True
                }
        
        # Default: return first from last results
        if self.session.last_recommendations:
            return {
                'recommendations': [self.session.last_recommendations[0]],
                'is_detail': True
            }
        
        return {
            'recommendations': [],
            'message': 'Which restaurant would you like to know more about?'
        }
    
    def handle_comparison(self, intent_result: IntentResult) -> Dict:
        """Handle comparison between restaurants."""
        # Try to extract two restaurant references
        query_lower = intent_result.constraints.occasion or ""  # Hacky but works
        
        # Get first two from last results if no specific references
        if self.session.last_recommendations and len(self.session.last_recommendations) >= 2:
            rec_a = self.session.last_recommendations[0]
            rec_b = self.session.last_recommendations[1]
            
            comparison_text = self.ranking_engine.generate_comparison(rec_a, rec_b)
            
            return {
                'recommendations': [rec_a, rec_b],
                'is_comparison': True,
                'comparison_text': comparison_text
            }
        
        return {
            'recommendations': [],
            'message': 'Please specify which restaurants to compare.'
        }
    
    def format_response(self, result: Dict, intent_result: IntentResult) -> str:
        """Format results as natural language response."""
        recommendations = result.get('recommendations', [])
        
        if not recommendations:
            return result.get('message', "I couldn't find any matching restaurants.")
        
        # Handle clarification
        if result.get('needs_clarification'):
            return self.generate_clarification_question(intent_result.constraints)
        
        # Handle comparison
        if result.get('is_comparison'):
            return result.get('comparison_text', 'Unable to generate comparison.')
        
        # Handle detail request
        if result.get('is_detail'):
            return self.format_restaurant_detail(recommendations[0])
        
        # Handle regular recommendations
        return self.format_recommendations(result, intent_result)
    
    def format_recommendations(self, result: Dict, intent_result: IntentResult) -> str:
        """Format recommendation list."""
        recommendations = result['recommendations']
        timing = result.get('timing_ms', 0)
        from_cache = result.get('from_cache', False)
        
        lines = []
        
        # Header
        if from_cache:
            lines.append(f" *Refined from cache ({timing:.0f}ms)*\n")
        
        if result.get('fallback_used'):
            lines.append(f" {result.get('fallback_reason')}\n")
        
        lines.append("Here are my top recommendations:\n")
        
        # Each restaurant
        for i, rec in enumerate(recommendations[:3]):
            r = rec.restaurant
            emoji = "" if i == 0 else "" if i == 1 else ""
            
            lines.append(f"\n{emoji} **{r.name}** (Confidence: {rec.confidence}%)")
            lines.append(f" {r.location} |  {r.cuisines[:50]}...")
            lines.append(f" {r.rating}/5 ({r.votes} reviews) |  ₹{r.approx_cost} for two")
            
            if rec.explanation:
                lines.append(f"\n{rec.explanation}")
            
            if rec.popular_dishes:
                dishes = ', '.join(rec.popular_dishes[:3])
                lines.append(f"\n *Popular: {dishes}*")
        
        # Follow-up suggestions
        lines.append("\n---")
        lines.append(self.generate_followup_suggestions(result, intent_result.constraints))
        
        return '\n'.join(lines)
    
    def format_restaurant_detail(self, rec: RankedRestaurant) -> str:
        """Format detailed restaurant info."""
        r = rec.restaurant
        
        lines = [
            f"# {r.name}\n",
            f" **Location**: {r.location}",
        ]
        
        # Add full address if available
        if r.address:
            lines.append(f"� **Address**: {r.address}")
        
        # Add phone if available
        if r.phone:
            lines.append(f" **Phone**: {r.phone}")
        
        lines.extend([
            f"� **Cuisines**: {r.cuisines}",
            f" **Rating**: {r.rating}/5 ({r.votes} reviews)",
            f" **Cost**: ₹{r.approx_cost} for two",
        ])
        
        if r.rest_type:
            lines.append(f" **Type**: {r.rest_type}")
        
        if r.online_order:
            lines.append(" Online ordering available")
        if r.book_table:
            lines.append(" Table booking available")
        
        if rec.popular_dishes:
            lines.append(f"\n **Popular Dishes**: {', '.join(rec.popular_dishes)}")
        
        if rec.explanation:
            lines.append(f"\n{rec.explanation}")
        
        return '\n'.join([l for l in lines if l])
    
    def generate_clarification_question(self, constraints: Constraints) -> str:
        """Generate clarifying question."""
        if not constraints.location:
            return "Which area of Bangalore are you looking for? (e.g., Koramangala, Indiranagar)"
        if not constraints.cuisines:
            return "Any specific cuisine you're craving? (e.g., Italian, North Indian, Chinese)"
        if not constraints.budget_max:
            return "What's your budget for two? (e.g., under ₹1000)"
        return "Could you tell me more about what you're looking for?"
    
    def generate_followup_suggestions(self, result: Dict, constraints: Constraints) -> str:
        """Generate proactive suggestions."""
        suggestions = []
        
        if not constraints.budget_max:
            suggestions.append(" Try: 'Show options under ₹1000'")
        
        if constraints.budget_max and len(result.get('recommendations', [])) > 0:
            recommendations = result['recommendations']
            if all(r.restaurant.approx_cost < constraints.budget_max * 0.7 for r in recommendations[:3]):
                suggestions.append(" Want to see slightly more upscale options?")
        
        if not any(r.restaurant.book_table for r in result.get('recommendations', [])[:3]):
            suggestions.append(" Filter: 'with table booking'")
        
        suggestions.append(" Ask: 'Tell me more about the first one'")
        suggestions.append(" Compare: 'Compare the first two'")
        
        return " **You might ask:**\n" + '\n'.join(f"  • {s}" for s in suggestions[:3])
    
    def process(self, query: str) -> str:
        """
        Main entry point. Process a user query and return response.
        
        Args:
            query: User's natural language query
            
        Returns:
            Natural language response string
        """
        start_time = time.time()
        
        # Step 1: Classify intent
        intent_result = self.classify_intent(query)
        print(f"\n Intent: {intent_result.intent.value} → Route: {intent_result.route.value}")
        
        # Step 2: Route to agents
        result = self.route_to_agents(intent_result)
        
        # Step 3: Format response
        response = self.format_response(result, intent_result)
        
        # Step 4: Update session
        total_time = (time.time() - start_time) * 1000
        turn = ConversationTurn(
            turn_id=self.session.turn_count + 1,
            timestamp=datetime.now(),
            user_query=query,
            intent=intent_result.intent,
            route=intent_result.route,
            constraints_used=intent_result.constraints,
            results=[rec.to_dict() for rec in result.get('recommendations', [])],
            response=response,
            timing_ms=total_time
        )
        self.session.add_turn(turn)
        
        print(f" Total time: {total_time:.0f}ms")
        
        return response
    
    def get_session_stats(self) -> Dict:
        """Get session statistics."""
        return self.session.get_stats()
    
    def reset_session(self):
        """Reset session state."""
        self.session.reset()
        self.session_rag.clear_session(self.session.session_id)
        print(" Session reset")


# Test function
def test_orchestrator():
    """Test the conversational orchestrator."""
    print("\n" + "=" * 60)
    print("TESTING CONVERSATIONAL ORCHESTRATOR")
    print("=" * 60)
    
    orchestrator = ConversationalOrchestrator()
    orchestrator.initialize()
    
    # Test queries
    queries = [
        "Find Italian restaurants in Koramangala under ₹1500",
        "Show me cheaper options under ₹800",
        "Tell me more about the first one",
        "Compare the first two"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"USER: {query}")
        print("=" * 60)
        response = orchestrator.process(query)
        print(f"\nASSISTANT:\n{response[:500]}...")
    
    # Print stats
    print(f"\n Session Stats: {orchestrator.get_session_stats()}")


if __name__ == "__main__":
    test_orchestrator()
