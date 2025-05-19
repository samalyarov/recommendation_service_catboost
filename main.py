#!/usr/bin/env python
# coding: utf-8

"""
Recommendation Service API

This module implements a FastAPI-based recommendation service that provides
personalized post recommendations to users based on their characteristics
and interaction history. The service uses CatBoost models for predictions
and implements A/B testing functionality.

Author: Sam Maliarov
Date: 2025-02-03
"""

# Library imports
import os
import hashlib
import logging
import time
from typing import List, Dict, Any, Tuple
from datetime import datetime
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from catboost import CatBoostClassifier
from pydantic import BaseModel, Field
from loguru import logger

# Configuration
class Config:
    """Configuration settings for the recommendation service."""
    SALT: str = os.getenv("SALT", "random_salt_value")
    SPLIT_PERCENTAGE: int = 50
    CHUNKSIZE: int = 100000
    DEFAULT_RECOMMENDATION_LIMIT: int = 5

# Database setup
Base = declarative_base()

# Database connection
def get_database_url() -> str:
    """Get database connection URL from environment variables."""
    database = os.getenv("DATABASE")
    user = os.getenv("USER")
    password = os.getenv("PASSWORD")
    host = os.getenv("HOST")
    port = os.getenv("PORT")
    
    if not all([database, user, password, host, port]):
        raise ValueError("Missing required database environment variables")
        
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"

SQLALCHEMY_DATABASE_URL = get_database_url()
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Pydantic models
class PostGet(BaseModel):
    """Model for post data in API responses."""
    id: int = Field(..., description="Unique identifier of the post")
    text: str = Field(..., description="Content of the post")
    topic: str = Field(..., description="Topic category of the post")

    class Config:
        orm_mode = True

class Response(BaseModel):
    """Model for API response."""
    exp_group: str = Field(..., description="Experiment group of the user")
    recommendations: List[PostGet] = Field(..., description="List of recommended posts")

# Model loading functions
def get_model_path(model_version: str) -> str:
    """
    Get the path to the model file.
    
    Args:
        model_version: Version identifier of the model ('control' or 'test')
        
    Returns:
        str: Path to the model file
    """
    if os.environ.get("IS_LMS") == "1":
        return f"/workdir/user_input/{model_version}" # when working in LMS - this was a study project after all
    return f"models/{model_version}"

def load_models() -> Tuple[CatBoostClassifier, CatBoostClassifier]:
    """
    Load both control and test models.
    
    Returns:
        Tuple[CatBoostClassifier, CatBoostClassifier]: Control and test models
    """
    logger.info('Loading the models...')
    
    model_control = CatBoostClassifier()
    model_control.load_model(get_model_path('model_control'))
    
    model_test = CatBoostClassifier()
    model_test.load_model(get_model_path('model_test'))
    
    return model_control, model_test

# Data loading functions
def batch_load_sql(query: str) -> pd.DataFrame:
    """
    Load data from SQL database in batches.
    
    Args:
        query: SQL query to execute
        
    Returns:
        pd.DataFrame: Combined dataframe from all batches
    """
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    
    try:
        for chunk_dataframe in pd.read_sql(query, conn, chunksize=Config.CHUNKSIZE):
            chunks.append(chunk_dataframe)
    finally:
        conn.close()
        
    return pd.concat(chunks, ignore_index=True)

def load_features() -> pd.DataFrame:
    """Load user-level features from the database."""
    logger.info('Loading the user-level features...')
    return batch_load_sql('SELECT * FROM recommendation_service_features')

def load_post_data() -> pd.DataFrame:
    """Load and enrich post-level data from the database."""
    logger.info('Loading the post-level data...')
    
    # Load base post data
    df = batch_load_sql('SELECT * FROM public.post_text_df')
    df['post_length'] = df['text'].apply(len)
    
    # Load additional post features
    logger.info('Loading the data for post information enrichment...')
    additional_post_features = batch_load_sql("""
        SELECT 
            post_id,
            COUNT(DISTINCT(user_id)) AS unique_user_interactions,
            COUNT(DISTINCT(user_id)) FILTER (WHERE action ='like') AS user_likes,
            COUNT(action) FILTER (WHERE action = 'view') AS total_post_views
        FROM public.feed_data
        GROUP BY post_id
    """)
    
    # Merge and calculate additional metrics
    df = df.merge(additional_post_features, on='post_id', how='left')
    df['post_likability'] = df['user_likes'] / df['unique_user_interactions']
    
    # Fill missing values
    for col in ['unique_user_interactions', 'user_likes', 'total_post_views', 'post_likability']:
        df[col] = df[col].fillna(0)
        
    return df

def get_exp_group(user_id: int) -> str:
    """
    Determine the experiment group for a user.
    
    Args:
        user_id: User identifier
        
    Returns:
        str: Experiment group ('control' or 'test')
    """
    user_id = abs(user_id)
    hash_object = hashlib.md5(f"{user_id}{Config.SALT}".encode())
    hash_int = int(hash_object.hexdigest(), 16)
    
    return 'control' if hash_int % 100 < Config.SPLIT_PERCENTAGE else 'test'

# Database session dependency
def get_db():
    """Dependency for database session management."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize FastAPI app
app = FastAPI(
    title="Recommendation Service API",
    description="API for providing personalized post recommendations",
    version="1.0.0"
)

# Load models and data at startup
model_control, model_test = load_models()
user_data = load_features()
post_data = load_post_data()

@app.get("/post/recommendations/", response_model=Response)
async def recommended_posts(
    id: int,
    limit: int = Config.DEFAULT_RECOMMENDATION_LIMIT,
    db: Session = Depends(get_db)
) -> Response:
    """
    Get personalized post recommendations for a user.
    
    Args:
        id: User identifier
        limit: Maximum number of recommendations to return
        db: Database session
        
    Returns:
        Response: Recommended posts and experiment group
        
    Raises:
        HTTPException: If user not found or other errors occur
    """
    start_time = time.time()
    
    try:
        # Get user data
        user_data_curr = user_data.loc[user_data['user_id'] == id]
        if user_data_curr.empty:
            raise HTTPException(status_code=404, detail="User not found")
        user_data_curr = user_data_curr.iloc[0]
        
        # Prepare user features
        user_info = {
            'gender': user_data_curr['gender'],
            'age': user_data_curr['age'],
            'country': user_data_curr['country'],
            'city': user_data_curr['city'],
            'exp_group': user_data_curr['exp_group'],
            'os': user_data_curr['os'],
            'source': user_data_curr['source'],
            'unique_post_interactions': user_data_curr['unique_post_interactions'],
            'posts_liked': user_data_curr['posts_liked'],
            'total_views': user_data_curr['total_views'],
            'posts_liked_share': user_data_curr['posts_liked_share']
        }
        
        # Prepare post data with user features
        local_post_data = post_data.assign(**user_info)
        now = datetime.now()
        local_post_data['month'] = now.month
        local_post_data['day'] = now.day
        local_post_data['weekday'] = now.weekday()
        local_post_data['hour'] = now.hour
        
        # Select features for prediction
        features = [
            'gender', 'age', 'country', 'city', 'exp_group', 'os', 'source',
            'unique_post_interactions', 'posts_liked', 'total_views', 'posts_liked_share',
            'topic', 'post_length', 'unique_user_interactions', 'user_likes',
            'total_post_views', 'post_likability',
            'month', 'day', 'weekday', 'hour'
        ]
        curr_post_data = local_post_data[features]
        
        # Make predictions
        group = get_exp_group(id)
        model = model_control if group == 'control' else model_test
        curr_post_data['pred'] = model.predict_proba(curr_post_data)[:, 1]
        
        # Get top recommendations
        top_idx = curr_post_data.nlargest(limit, 'pred').index
        top_posts = local_post_data.loc[top_idx, ['post_id', 'text', 'topic']]
        
        # Format response
        recommendations = [
            PostGet(id=row['post_id'], text=row['text'], topic=row['topic'])
            for _, row in top_posts.iterrows()
        ]
        
        logger.info(f"Request completed in {time.time() - start_time:.2f} seconds")
        return Response(exp_group=group, recommendations=recommendations)
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
