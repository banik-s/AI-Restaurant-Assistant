#!/usr/bin/env python3
"""
Run Script - Executes the complete data pipeline
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    print("=" * 60)
    print("  MULTI-AGENT RESTAURANT DISCOVERY SYSTEM")
    print("=" * 60)
    print("\nThis script will run the complete data pipeline.\n")
    
    # Check for dependencies
    print(" Checking dependencies...")
    try:
        import pandas
        import numpy
        import chromadb
        import sentence_transformers
        import openai
        print("    All core dependencies installed")
    except ImportError as e:
        print(f"    Missing dependency: {e}")
        print("\n   Please run: pip install -r requirements.txt")
        return
    
    # Step 1: Data Preprocessing
    print("\n" + "=" * 60)
    print("STEP 1: DATA PREPROCESSING")
    print("=" * 60)
    
    from pipelines.data_preprocessing import preprocess_data
    from config.settings import SAMPLE_SIZE, PROCESSED_DATA_DIR
    
    processed_path = PROCESSED_DATA_DIR / "restaurants_processed.parquet"
    
    if processed_path.exists():
        print(f" Processed data already exists at: {processed_path}")
        response = input("   Reprocess? (y/N): ").strip().lower()
        if response != 'y':
            print("   Skipping preprocessing...")
            df = pandas.read_parquet(processed_path)
        else:
            df = preprocess_data(sample_size=SAMPLE_SIZE)
    else:
        df = preprocess_data(sample_size=SAMPLE_SIZE)
    
    # Step 2: Feature Extraction
    print("\n" + "=" * 60)
    print("STEP 2: FEATURE EXTRACTION")
    print("=" * 60)
    
    from pipelines.feature_extraction import extract_features
    
    features_path = PROCESSED_DATA_DIR / "restaurants_with_features.parquet"
    
    if features_path.exists():
        print(f" Features already exist at: {features_path}")
        response = input("   Re-extract? (y/N): ").strip().lower()
        if response != 'y':
            print("   Skipping feature extraction...")
            df = pandas.read_parquet(features_path)
        else:
            df = extract_features(df, use_trained_model=True)
    else:
        df = extract_features(df, use_trained_model=True)
    
    # Step 3: Embedding and Vector Store
    print("\n" + "=" * 60)
    print("STEP 3: EMBEDDING & VECTOR STORE")
    print("=" * 60)
    
    from pipelines.embedding_pipeline import create_embeddings_and_store
    from config.settings import CHROMA_DB_DIR
    
    chroma_exists = (CHROMA_DB_DIR / "chroma.sqlite3").exists()
    
    if chroma_exists:
        print(f" ChromaDB already exists at: {CHROMA_DB_DIR}")
        response = input("   Recreate? (y/N): ").strip().lower()
        if response != 'y':
            print("   Skipping embedding generation...")
        else:
            create_embeddings_and_store(df)
    else:
        create_embeddings_and_store(df)
    
    # Done!
    print("\n" + "=" * 60)
    print(" PIPELINE COMPLETE!")
    print("=" * 60)
    print("\n To start the application, run:")
    print("   streamlit run ui/streamlit_app.py")
    print("\n")


if __name__ == "__main__":
    main()
