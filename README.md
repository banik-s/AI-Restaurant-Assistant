# AI-Powered Restaurant Discovery System

An intelligent, conversational restaurant recommendation system for Bangalore that understands natural language queries and provides personalized, explainable recommendations using multi-agent AI architecture.

## Features

- ** Natural Language Understanding**: Ask in plain English - "Find romantic Italian restaurants in Koramangala under ₹1500"
- ** Multi-Agent Architecture**: Specialized AI agents working together for accurate recommendations
- ** Conversational AI**: Multi-turn dialogue with context awareness - refine, compare, and ask follow-ups
- ** Intelligent Ranking**: Sentiment analysis + aspect-based scoring + intent alignment
- ** Explainable Recommendations**: Every suggestion backed by review analysis and reasoning
- ** Fast & Efficient**: Session-scoped indexing and smart caching for instant responses
- ** Occasion-Aware**: Understands context like anniversaries, business meetings, casual hangouts

##  Architecture

### Multi-Agent System

**Agent 1: Conversational Orchestrator**
- Manages conversation flow and context
- Classifies user intent (new search, refinement, comparison)
- Routes queries intelligently to other agents
- Learns user preferences across conversation

**Agent 2: SQL Query Builder & Data Fetcher**
- Translates natural language to optimized SQL
- Fetches relevant restaurants from 51K+ database
- Smart filtering with fallback strategies
- Returns only necessary data (50-100 results)

**Agent 3: Session-Scoped RAG Agent**
- Creates temporary vector index from fetched results
- Enables semantic search within current session
- Handles refinements without database hits
- Lightning-fast comparisons and follow-up queries

**Agent 4: Ranking & Recommendation Engine**
- Fine-tuned DistilBERT for sentiment analysis
- Aspect-based scoring (food, service, ambiance, value)
- Intent-aligned ranking with explainability
- Generates natural language recommendations

##  Quick Start

### Prerequisites

```bash
Python 3.10+
pip install -r requirements.txt
Add openai api key in settings file
