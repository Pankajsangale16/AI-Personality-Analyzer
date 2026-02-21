#!/usr/bin/env python3
"""
Quick training script for personality prediction models
"""

import os
import sys
import pandas as pd
import numpy as np
from django.conf import settings

# Set up Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'personality_ai.settings')
import django
django.setup()

from predictor.ml_model import PersonalityPredictor

def create_training_data():
    """Create properly balanced training data"""
    
    # Simple, balanced training data
    training_data = {
        'text': [
            # High Openness
            "I love exploring new ideas and trying different things every day. Creativity is my passion.",
            "Art and music inspire me deeply. I enjoy philosophical discussions and abstract thinking.",
            
            # Low Openness  
            "I prefer routine and structure in my daily life. Change makes me uncomfortable.",
            "I stick to what I know works best. Practical solutions are more important than creative ones.",
            
            # High Conscientiousness
            "I always plan my schedule carefully and stick to my goals no matter what.",
            "I'm very detail-oriented and make sure everything is perfect before completing a task.",
            
            # Low Conscientiousness
            "I'm more spontaneous and flexible with my plans. I prefer to go with the flow.",
            "I don't worry too much about details. I focus on the big picture and figure things out.",
            
            # High Extraversion
            "I enjoy being around people and going to social events every weekend.",
            "I love meeting new people and making friends. Social interactions energize me.",
            
            # Low Extraversion
            "I work best alone and need quiet time to recharge my energy.",
            "I prefer deep conversations with close friends over small talk with many people.",
            
            # High Agreeableness
            "I always try to help others and maintain harmony in all my relationships.",
            "I'm very empathetic and care deeply about others' feelings. Kindness is my principle.",
            
            # Low Agreeableness
            "I'm very competitive and always strive to be the best. I'm not afraid of confrontation.",
            "I prioritize my goals over being nice to others. Sometimes you have to be tough.",
            
            # High Neuroticism
            "I often feel worried and stressed about small things in life. Anxiety affects me.",
            "I'm very sensitive and my emotions change quickly. I get upset easily.",
            
            # Low Neuroticism
            "I'm very easy-going and don't let things bother me too much.",
            "I stay calm under pressure and handle stress well. I'm emotionally stable."
        ],
        'openness': [0.9, 0.8, 0.2, 0.3, 0.5, 0.6, 0.4, 0.3, 0.7, 0.8, 0.2, 0.3, 0.6, 0.7, 0.3, 0.4, 0.9, 0.8, 0.2, 0.3],
        'conscientiousness': [0.4, 0.5, 0.8, 0.7, 0.9, 0.85, 0.3, 0.35, 0.6, 0.5, 0.4, 0.6, 0.7, 0.8, 0.3, 0.4, 0.5, 0.4, 0.8, 0.7],
        'extraversion': [0.6, 0.5, 0.4, 0.3, 0.5, 0.6, 0.7, 0.6, 0.9, 0.85, 0.2, 0.25, 0.7, 0.6, 0.8, 0.7, 0.4, 0.3, 0.6, 0.5],
        'agreeableness': [0.7, 0.8, 0.6, 0.5, 0.6, 0.7, 0.5, 0.4, 0.8, 0.9, 0.7, 0.6, 0.9, 0.85, 0.3, 0.35, 0.4, 0.3, 0.7, 0.8],
        'neuroticism': [0.3, 0.4, 0.5, 0.4, 0.2, 0.3, 0.6, 0.5, 0.4, 0.3, 0.6, 0.5, 0.3, 0.4, 0.9, 0.85, 0.2, 0.3, 0.8, 0.7]
    }
    
    df = pd.DataFrame(training_data)
    
    # Save to CSV
    os.makedirs('training_data', exist_ok=True)
    df.to_csv('training_data/personality_data.csv', index=False)
    print(f"✅ Created training data with {len(df)} samples")
    
    return df

def main():
    """Main training function"""
    print("🚀 Quick Training Personality Models...")
    
    # Create training data
    df = create_training_data()
    
    # Display data statistics
    print("\n📊 Training Data Summary:")
    print(df.describe())
    
    # Initialize predictor
    print("\n🤖 Initializing personality predictor...")
    predictor = PersonalityPredictor()
    
    # Train models
    print("\n📚 Training personality prediction models...")
    try:
        predictor.train_models('training_data/personality_data.csv')
        print("✅ Models trained successfully!")
        
        # Test prediction
        print("\n🧪 Testing prediction...")
        test_text = "I love exploring new ideas and meeting new people. I'm creative and organized."
        scores = predictor.predict_personality(test_text)
        
        print("Prediction results:")
        for trait, score in scores.items():
            print(f"  {trait.capitalize()}: {score:.3f}")
        
        print("\n🎉 Training completed successfully!")
        print("Models are ready for use in the web application.")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
