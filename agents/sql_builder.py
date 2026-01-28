
import sys
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import METADATA_DB_PATH
from models.schemas import Constraints, Restaurant, SQLQueryResult


class SQLQueryBuilder:
    """
    Agent 2: Intelligent database interface.
    
    Responsibilities:
    - Build optimized SQL from constraints
    - Execute on SQLite (fast indexed queries)
    - Fetch minimal data (50-100 restaurants max)
    - Smart fallbacks when no exact matches
    """
    
    def __init__(self):
        self.db_path = METADATA_DB_PATH
        self.default_limit = 50
        self.default_min_rating = 3.0
        
        # Known locations and cuisines (for fuzzy matching)
        self._known_locations: List[str] = []
        self._known_locations_lower: Dict[str, str] = {}
        self._known_cuisines: List[str] = []
        self._known_cuisines_lower: Dict[str, str] = {}
    
    def initialize(self):
        print(" Initializing SQL Query Builder...")
        
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Database not found at {self.db_path}. "
                "Run: python pipelines/setup_database.py"
            )
        
        self.load_known_values()
        print(f"   Database: {self.db_path}")
        print(f"   Locations: {len(self._known_locations)}, Cuisines: {len(self._known_cuisines)}")
        return self
    
    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    
    def load_known_values(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Load locations
        cursor.execute("SELECT location FROM locations ORDER BY count DESC")
        self._known_locations = [row[0] for row in cursor.fetchall()]
        self._known_locations_lower = {loc.lower(): loc for loc in self._known_locations}
        
        # Load cuisines
        cursor.execute("SELECT cuisine FROM cuisines ORDER BY count DESC")
        self._known_cuisines = [row[0] for row in cursor.fetchall()]
        self._known_cuisines_lower = {c.lower(): c for c in self._known_cuisines}
        
        conn.close()
    
    def normalize_location(self, location: str) -> Optional[str]:
        if not location:
            return None
        
        location_lower = location.lower().strip()
        
        if location_lower in self._known_locations_lower:
            return self._known_locations_lower[location_lower]
        
        # Partial match
        for known_lower, known in self._known_locations_lower.items():
            if location_lower in known_lower or known_lower in location_lower:
                return known
        
        try:
            from rapidfuzz import process, fuzz
            result = process.extractOne(
                location_lower,
                list(self._known_locations_lower.keys()),
                scorer=fuzz.ratio,
                score_cutoff=70
            )
            if result:
                return self._known_locations_lower[result[0]]
        except ImportError:
            pass
        
        return None
    
    def normalize_cuisine(self, cuisine: str) -> Optional[str]:
        """Normalize cuisine to canonical form."""
        if not cuisine:
            return None
        
        cuisine_lower = cuisine.lower().strip()
        
        # Exact match
        if cuisine_lower in self._known_cuisines_lower:
            return self._known_cuisines_lower[cuisine_lower]
        
        # Partial match
        for known_lower, known in self._known_cuisines_lower.items():
            if cuisine_lower in known_lower or known_lower in cuisine_lower:
                return known
        
        return None
    
    def build_query(self, constraints: Constraints, limit: int = None) -> Tuple[str, List, Dict]:
        
        limit = limit or self.default_limit
        conditions = []
        params = []
        filters_applied = {}
        
        # Location filter (most important for indexing)
        if constraints.location:
            normalized = self.normalize_location(constraints.location)
            if normalized:
                conditions.append("location_lower = ?")
                params.append(normalized.lower())
                filters_applied['location'] = normalized
        elif constraints.locations:
            normalized_locs = [self.normalize_location(loc) for loc in constraints.locations]
            valid_locs = [(loc, loc.lower()) for loc in normalized_locs if loc]
            if valid_locs:
                placeholders = ','.join(['?' for _ in valid_locs])
                conditions.append(f"location_lower IN ({placeholders})")
                params.extend([loc[1] for loc in valid_locs])
                filters_applied['locations'] = [loc[0] for loc in valid_locs]
        
        # Cuisine filter (LIKE for flexibility)
        if constraints.cuisines:
            cuisine_conditions = []
            applied_cuisines = []
            for cuisine in constraints.cuisines:
                normalized = self.normalize_cuisine(cuisine)
                if normalized:
                    cuisine_conditions.append("cuisines_lower LIKE ?")
                    params.append(f"%{normalized.lower()}%")
                    applied_cuisines.append(normalized)
                else:
                    # Try raw match
                    cuisine_conditions.append("cuisines_lower LIKE ?")
                    params.append(f"%{cuisine.lower()}%")
                    applied_cuisines.append(cuisine)
            
            if cuisine_conditions:
                conditions.append(f"({' OR '.join(cuisine_conditions)})")
                filters_applied['cuisines'] = applied_cuisines
        
        # Dish filter (searches BOTH cuisines and dish_liked columns)
        # This is because dishes like 'biryani' often appear in cuisines column
        if constraints.dishes:
            dish_conditions = []
            for dish in constraints.dishes:
                # Search in both cuisines and dish_liked
                dish_conditions.append("(cuisines_lower LIKE ? OR LOWER(COALESCE(dish_liked, '')) LIKE ?)")
                params.append(f"%{dish.lower()}%")
                params.append(f"%{dish.lower()}%")
            
            if dish_conditions:
                conditions.append(f"({' OR '.join(dish_conditions)})")
                filters_applied['dishes'] = constraints.dishes
        
        # Budget filter
        if constraints.budget_max:
            conditions.append("approx_cost <= ?")
            params.append(constraints.budget_max)
            filters_applied['budget_max'] = constraints.budget_max
        
        if constraints.budget_min:
            conditions.append("approx_cost >= ?")
            params.append(constraints.budget_min)
            filters_applied['budget_min'] = constraints.budget_min
        
        min_rating = constraints.min_rating or self.default_min_rating
        conditions.append("rating >= ?")
        params.append(min_rating)
        filters_applied['min_rating'] = min_rating
        
        if constraints.online_order is not None:
            conditions.append("online_order = ?")
            params.append(1 if constraints.online_order else 0)
            filters_applied['online_order'] = constraints.online_order
        
        if constraints.book_table is not None:
            conditions.append("book_table = ?")
            params.append(1 if constraints.book_table else 0)
            filters_applied['book_table'] = constraints.book_table
        
        # Build final query - GROUP BY name to avoid showing same restaurant multiple times
        sql = """
            SELECT 
                restaurant_id as id,
                name,
                location,
                cuisines,
                approx_cost,
                rating,
                votes,
                online_order,
                book_table,
                rest_type,
                phone,
                address,
                dish_liked,
                sentiment_score,
                food_quality_score,
                service_score,
                ambiance_score
            FROM restaurants_metadata
        """
        
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        
        # Group by name to deduplicate (keep highest rated branch)  
        sql += " GROUP BY name"
        sql += " ORDER BY rating DESC, votes DESC"
        sql += f" LIMIT {limit}"
        
        return sql, params, filters_applied
    
    def execute(self, constraints: Constraints, limit: int = None) -> SQLQueryResult:
        start_time = time.time()
        limit = limit or self.default_limit
        
        # Build and execute query
        sql, params, filters_applied = self.build_query(constraints, limit)
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
        except Exception as e:
            print(f" SQL Error: {e}")
            rows = []
        finally:
            conn.close()
        
        # Convert to Restaurant objects
        restaurants = []
        for row in rows:
            restaurants.append(Restaurant(
                id=row['id'],
                name=row['name'],
                location=row['location'],
                cuisines=row['cuisines'],
                approx_cost=row['approx_cost'],
                rating=row['rating'],
                votes=row['votes'],
                online_order=bool(row['online_order']),
                book_table=bool(row['book_table']),
                rest_type=row['rest_type'] or "",
                phone=row['phone'] or "",
                address=row['address'] or "",
                dish_liked=row['dish_liked'] or ""
            ))
        
        execution_time = (time.time() - start_time) * 1000
        
        return SQLQueryResult(
            restaurants=restaurants,
            count=len(restaurants),
            filters_applied=filters_applied,
            execution_time_ms=execution_time,
            fallback_used=False
        )
    
    def execute_with_fallback(self, constraints: Constraints, limit: int = None) -> SQLQueryResult:
        result = self.execute(constraints, limit)
        
        if result.count > 0:
            return result
        
        # Fallback 1: Relax cuisine filter
        if constraints.cuisines:
            relaxed = Constraints(
                location=constraints.location,
                budget_max=constraints.budget_max,
                min_rating=constraints.min_rating
            )
            result = self.execute(relaxed, limit)
            if result.count > 0:
                result.fallback_used = True
                result.fallback_reason = f"No exact {', '.join(constraints.cuisines)} match, showing all cuisines"
                return result
        
        # Fallback 2: Relax location filter (nearby areas)
        if constraints.location:
            relaxed = Constraints(
                cuisines=constraints.cuisines,
                budget_max=constraints.budget_max,
                min_rating=constraints.min_rating
            )
            result = self.execute(relaxed, limit)
            if result.count > 0:
                result.fallback_used = True
                result.fallback_reason = f"No results in {constraints.location}, showing other areas"
                return result
        
        # Fallback 3: Increase budget by 20%
        if constraints.budget_max:
            increased_budget = int(constraints.budget_max * 1.2)
            relaxed = Constraints(
                location=constraints.location,
                cuisines=constraints.cuisines,
                budget_max=increased_budget,
                min_rating=constraints.min_rating
            )
            result = self.execute(relaxed, limit)
            if result.count > 0:
                result.fallback_used = True
                result.fallback_reason = f"No options under ₹{constraints.budget_max}, showing up to ₹{increased_budget}"
                return result
        
        relaxed = Constraints(min_rating=3.5)
        result = self.execute(relaxed, limit)
        if result.count > 0:
            result.fallback_used = True
            result.fallback_reason = "No exact match found, showing top-rated restaurants"
        
        return result
    
    def get_restaurant_by_id(self, restaurant_id: int) -> Optional[Restaurant]:
        """Get a single restaurant by ID."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT restaurant_id as id, name, location, cuisines, approx_cost,
                   rating, votes, online_order, book_table, rest_type, phone, address
            FROM restaurants_metadata
            WHERE restaurant_id = ?
        """, (restaurant_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return Restaurant(
                id=row['id'],
                name=row['name'],
                location=row['location'],
                cuisines=row['cuisines'],
                approx_cost=row['approx_cost'],
                rating=row['rating'],
                votes=row['votes'],
                online_order=bool(row['online_order']),
                book_table=bool(row['book_table']),
                rest_type=row['rest_type'] or "",
                phone=row['phone'] or "",
                address=row['address'] or ""
            )
        return None
    
    def get_known_locations(self) -> List[str]:
        """Get list of known locations."""
        return self._known_locations
    
    def get_known_cuisines(self) -> List[str]:
        """Get list of known cuisines."""
        return self._known_cuisines


# Test function
def test_sql_builder():
    
    builder = SQLQueryBuilder()
    builder.initialize()
    
    # Test 1: Basic query
    print("\n Test 1: Italian in Koramangala under ₹1500")
    constraints = Constraints(
        location="koramangala",
        cuisines=["Italian"],
        budget_max=1500
    )
    result = builder.execute(constraints)
    print(f"   Found: {result.count} restaurants in {result.execution_time_ms:.1f}ms")
    print(f"   Filters: {result.filters_applied}")
    for r in result.restaurants[:3]:
        print(f"   - {r.name} | {r.cuisines[:30]}... | ★{r.rating} | ₹{r.approx_cost}")
    
    # Test 2: With fallback
    print("\n Test 2: Sushi in BTM (likely no results)")
    constraints = Constraints(
        location="btm",
        cuisines=["Sushi", "Japanese"],
        budget_max=1000
    )
    result = builder.execute_with_fallback(constraints)
    print(f"   Found: {result.count} restaurants")
    print(f"   Fallback used: {result.fallback_used}")
    if result.fallback_reason:
        print(f"   Reason: {result.fallback_reason}")
    
    # Test 3: Location normalization
    print("\n Test 3: Location normalization")
    for loc in ["koramangla", "indiranagar", "btm layout", "HSR"]:
        normalized = builder.normalize_location(loc)
        print(f"   '{loc}' → '{normalized}'")


if __name__ == "__main__":
    test_sql_builder()
