"""
Agent 3: Session-Scoped RAG & Vector Search Agent
Creates temporary vector index from fetched data and enables semantic search over ONLY session restaurants
"""
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import EMBEDDING_MODEL_NAME
from models.schemas import Restaurant, RAGSearchResult


class SessionRAG:
    """
    Agent 3: Session-scoped vector search.
    
    Key Insight: Index ONLY the 50-100 restaurants fetched for this session.
    Not the full 51K dataset!
    
    Benefits:
    - Fast indexing: 50 restaurants in <1 second
    - No memory overhead from full dataset  
    - Accurate retrieval: only relevant restaurants
    - Session isolation: different users don't interfere
    """
    
    def __init__(self):
        self.embedding_model = None
        self.chroma_client = None
        
        # Session data stored in memory (not in ChromaDB for simplicity)
        self._session_restaurants: Dict[str, List[Restaurant]] = {}
        self._session_embeddings: Dict[str, Dict[int, List[float]]] = {}
    
    def initialize(self):
        """Initialize embedding model."""
        print(" Initializing Session RAG...")
        self._load_embedding_model()
        print(" Session RAG ready!")
        return self
    
    def _load_embedding_model(self):
        """Load the sentence-transformers embedding model."""
        from sentence_transformers import SentenceTransformer
        print(f"   Loading embedding model: {EMBEDDING_MODEL_NAME}")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    def _build_restaurant_text(self, restaurant: Restaurant) -> str:
        """
        Build rich text representation for embedding.
        
        Includes all semantic information for better matching.
        """
        parts = [
            f"Restaurant: {restaurant.name}",
            f"Location: {restaurant.location}",
            f"Cuisines: {restaurant.cuisines}",
            f"Type: {restaurant.rest_type}" if restaurant.rest_type else "",
            f"Rating: {restaurant.rating}/5 with {restaurant.votes} reviews",
            f"Cost: {restaurant.approx_cost} rupees for two",
        ]
        
        # Add features
        features = []
        if restaurant.online_order:
            features.append("online ordering available")
        if restaurant.book_table:
            features.append("table booking available")
        if features:
            parts.append(f"Features: {', '.join(features)}")
        
        # Add popular dishes if available
        if restaurant.dish_liked:
            parts.append(f"Popular dishes: {restaurant.dish_liked}")
        
        return ". ".join([p for p in parts if p])
    
    def index_restaurants(
        self, 
        restaurants: List[Restaurant], 
        session_id: str
    ) -> float:
        """
        Create embeddings for session restaurants.
        
        Args:
            restaurants: List of restaurants to index
            session_id: Unique session identifier
            
        Returns:
            Time taken in milliseconds
        """
        start_time = time.time()
        
        if not restaurants:
            return 0
        
        print(f" Indexing {len(restaurants)} restaurants for session {session_id[:8]}...")
        
        # Build text representations
        texts = [self._build_restaurant_text(r) for r in restaurants]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts)
        
        # Store in session data
        self._session_restaurants[session_id] = restaurants.copy()
        self._session_embeddings[session_id] = {
            r.id: embeddings[i].tolist() for i, r in enumerate(restaurants)
        }
        
        elapsed_ms = (time.time() - start_time) * 1000
        print(f"   Indexed in {elapsed_ms:.0f}ms")
        
        return elapsed_ms
    
    def semantic_search(
        self, 
        query: str, 
        session_id: str, 
        top_k: int = 10
    ) -> RAGSearchResult:
        """
        Semantic search within session restaurants.
        
        Args:
            query: User's semantic query
            session_id: Session to search in
            top_k: Number of results
            
        Returns:
            RAGSearchResult with matched restaurants and scores
        """
        if session_id not in self._session_restaurants:
            return RAGSearchResult(
                restaurants=[],
                source="session_cache",
                similarity_scores={}
            )
        
        restaurants = self._session_restaurants[session_id]
        embeddings = self._session_embeddings[session_id]
        
        if not restaurants:
            return RAGSearchResult(
                restaurants=[],
                source="session_cache",
                similarity_scores={}
            )
        
        # Generate query embedding
        import numpy as np
        query_embedding = self.embedding_model.encode(query)
        
        # Compute similarities
        scored_results = []
        for restaurant in restaurants:
            if restaurant.id not in embeddings:
                continue
            
            doc_embedding = np.array(embeddings[restaurant.id])
            query_emb = np.array(query_embedding)
            
            # Cosine similarity
            similarity = np.dot(query_emb, doc_embedding) / (
                np.linalg.norm(query_emb) * np.linalg.norm(doc_embedding)
            )
            
            scored_results.append((restaurant, float(similarity)))
        
        # Sort by similarity
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top K
        top_results = scored_results[:top_k]
        
        return RAGSearchResult(
            restaurants=[r for r, _ in top_results],
            source="session_cache",
            similarity_scores={r.id: score for r, score in top_results}
        )
    
    def filter_cached(
        self, 
        session_id: str, 
        budget_max: Optional[int] = None,
        budget_min: Optional[int] = None,
        min_rating: Optional[float] = None,
        cuisines: Optional[List[str]] = None,
        dietary: Optional[List[str]] = None,
        online_order: Optional[bool] = None,
        book_table: Optional[bool] = None
    ) -> RAGSearchResult:
        """
        Filter cached restaurants in-memory (instant!).
        
        This is the key optimization for refinement queries.
        No database hit needed.
        """
        if session_id not in self._session_restaurants:
            return RAGSearchResult(
                restaurants=[],
                source="session_cache",
                filter_applied="no_cache"
            )
        
        results = self._session_restaurants[session_id].copy()
        filter_desc = []
        
        # Apply filters
        if budget_max:
            results = [r for r in results if r.approx_cost <= budget_max]
            filter_desc.append(f"under ₹{budget_max}")
        
        if budget_min:
            results = [r for r in results if r.approx_cost >= budget_min]
            filter_desc.append(f"above ₹{budget_min}")
        
        if min_rating:
            results = [r for r in results if r.rating >= min_rating]
            filter_desc.append(f"rating >= {min_rating}")
        
        if cuisines:
            cuisine_lower = [c.lower() for c in cuisines]
            results = [
                r for r in results 
                if any(c in r.cuisines.lower() for c in cuisine_lower)
            ]
            filter_desc.append(f"cuisines: {', '.join(cuisines)}")
        
        if dietary:
            dietary_lower = [d.lower() for d in dietary]
            results = [
                r for r in results 
                if any(d in r.cuisines.lower() for d in dietary_lower)
            ]
            filter_desc.append(f"dietary: {', '.join(dietary)}")
        
        if online_order is not None:
            results = [r for r in results if r.online_order == online_order]
            filter_desc.append("online order" if online_order else "no online order")
        
        if book_table is not None:
            results = [r for r in results if r.book_table == book_table]
            filter_desc.append("table booking" if book_table else "no table booking")
        
        return RAGSearchResult(
            restaurants=results,
            source="session_cache",
            filter_applied=", ".join(filter_desc) if filter_desc else None
        )
    
    def get_by_id(self, session_id: str, restaurant_id: int) -> Optional[Restaurant]:
        """Get a specific restaurant from session cache."""
        if session_id not in self._session_restaurants:
            return None
        
        for r in self._session_restaurants[session_id]:
            if r.id == restaurant_id:
                return r
        return None
    
    def get_by_name(self, session_id: str, name: str) -> Optional[Restaurant]:
        """Get restaurant by name (fuzzy match)."""
        if session_id not in self._session_restaurants:
            return None
        
        name_lower = name.lower()
        for r in self._session_restaurants[session_id]:
            if name_lower in r.name.lower():
                return r
        return None
    
    def get_by_position(self, session_id: str, position: int, last_results: List[Restaurant]) -> Optional[Restaurant]:
        """Get restaurant by position in last results (1-indexed)."""
        if 1 <= position <= len(last_results):
            return last_results[position - 1]
        return None
    
    def get_all_cached(self, session_id: str) -> List[Restaurant]:
        """Get all cached restaurants for a session."""
        return self._session_restaurants.get(session_id, [])
    
    def has_cache(self, session_id: str) -> bool:
        """Check if session has cached restaurants."""
        return session_id in self._session_restaurants and len(self._session_restaurants[session_id]) > 0
    
    def clear_session(self, session_id: str):
        """Clear session cache."""
        if session_id in self._session_restaurants:
            del self._session_restaurants[session_id]
        if session_id in self._session_embeddings:
            del self._session_embeddings[session_id]
        print(f"  Cleared session cache: {session_id[:8]}")
    
    def get_session_stats(self, session_id: str) -> Dict:
        """Get statistics for a session."""
        count = len(self._session_restaurants.get(session_id, []))
        has_embeddings = session_id in self._session_embeddings
        return {
            'session_id': session_id,
            'cached_count': count,
            'embeddings_ready': has_embeddings
        }


# Test function
def test_session_rag():
    """Test the session RAG agent."""
    print("\n" + "=" * 60)
    print("TESTING SESSION RAG")
    print("=" * 60)
    
    # Initialize
    rag = SessionRAG()
    rag.initialize()
    
    # Create test restaurants
    test_restaurants = [
        Restaurant(id=1, name="Tandoor Palace", location="Koramangala",
                   cuisines="North Indian, Mughlai", approx_cost=900, rating=4.5, votes=200),
        Restaurant(id=2, name="Pizza Corner", location="Koramangala",
                   cuisines="Italian, Pizza, Fast Food", approx_cost=600, rating=4.0, votes=150),
        Restaurant(id=3, name="Thai Orchid", location="Koramangala",
                   cuisines="Thai, Asian", approx_cost=1200, rating=4.3, votes=180),
        Restaurant(id=4, name="Biryani House", location="Koramangala",
                   cuisines="Biryani, North Indian", approx_cost=500, rating=4.2, votes=300),
        Restaurant(id=5, name="Veggie Delight", location="Koramangala",
                   cuisines="Vegetarian, South Indian", approx_cost=400, rating=4.1, votes=100),
    ]
    
    session_id = "test_session_123"
    
    # Test 1: Index restaurants
    print("\n Test 1: Index restaurants")
    index_time = rag.index_restaurants(test_restaurants, session_id)
    print(f"   Indexed {len(test_restaurants)} restaurants in {index_time:.0f}ms")
    
    # Test 2: Semantic search
    print("\n Test 2: Semantic search - 'romantic dinner'")
    result = rag.semantic_search("romantic dinner with good ambiance", session_id, top_k=3)
    for r in result.restaurants:
        score = result.similarity_scores.get(r.id, 0)
        print(f"   - {r.name} | {r.cuisines[:20]}... | score={score:.3f}")
    
    # Test 3: Filter cached
    print("\n Test 3: Filter - under ₹700")
    result = rag.filter_cached(session_id, budget_max=700)
    print(f"   Found {len(result.restaurants)} restaurants")
    for r in result.restaurants:
        print(f"   - {r.name} | ₹{r.approx_cost}")
    
    # Test 4: Get by name
    print("\n Test 4: Get by name - 'biryani'")
    restaurant = rag.get_by_name(session_id, "biryani")
    if restaurant:
        print(f"   Found: {restaurant.name} | {restaurant.cuisines}")
    
    # Test 5: Session stats
    print("\n Session stats:")
    stats = rag.get_session_stats(session_id)
    print(f"   {stats}")


if __name__ == "__main__":
    test_session_rag()
