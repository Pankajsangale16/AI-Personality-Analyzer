from django.shortcuts import render, redirect
from django.http import JsonResponse
from .models import PersonalityPrediction
from .ml_model import PersonalityPredictor
import json

# Initialize the ML model
predictor = PersonalityPredictor()

def home(request):
    """Home page with input form"""
    return render(request, 'predictor/home.html')

def predict_personality(request):
    """Process text input and predict personality"""
    if request.method == 'POST':
        text_input = request.POST.get('text_input', '')
        twitter_handle = request.POST.get('twitter_handle', '')
        
        # If Twitter handle is provided, fetch tweets
        if twitter_handle:
            try:
                text_input = predictor.fetch_twitter_posts(twitter_handle)
            except Exception as e:
                return render(request, 'predictor/home.html', {
                    'error': f"Error fetching Twitter data: {str(e)}"
                })
        
        if not text_input.strip():
            return render(request, 'predictor/home.html', {
                'error': 'Please provide some text to analyze.'
            })
        
        # Predict personality
        try:
            scores = predictor.predict_personality(text_input)
            
            # Save to database
            prediction = PersonalityPrediction.objects.create(
                text_input=text_input[:500],  # Store first 500 chars
                twitter_handle=twitter_handle if twitter_handle else None,
                openness_score=scores['openness'],
                conscientiousness_score=scores['conscientiousness'],
                extraversion_score=scores['extraversion'],
                agreeableness_score=scores['agreeableness'],
                neuroticism_score=scores['neuroticism']
            )
            
            return redirect('result', prediction_id=prediction.id)
            
        except Exception as e:
            return render(request, 'predictor/home.html', {
                'error': f"Error during prediction: {str(e)}"
            })
    
    return redirect('home')

def result(request, prediction_id):
    """Display prediction results"""
    try:
        prediction = PersonalityPrediction.objects.get(id=prediction_id)
        
        # Prepare data for visualization
        scores = {
            'Openness': prediction.openness_score,
            'Conscientiousness': prediction.conscientiousness_score,
            'Extraversion': prediction.extraversion_score,
            'Agreeableness': prediction.agreeableness_score,
            'Neuroticism': prediction.neuroticism_score
        }
        
        # Calculate percentages for progress bars
        percentages = {
            'Openness': int(prediction.openness_score * 100),
            'Conscientiousness': int(prediction.conscientiousness_score * 100),
            'Extraversion': int(prediction.extraversion_score * 100),
            'Agreeableness': int(prediction.agreeableness_score * 100),
            'Neuroticism': int(prediction.neuroticism_score * 100)
        }
        
        # Generate descriptions
        descriptions = generate_descriptions(scores)
        
        context = {
            'prediction': prediction,
            'scores': scores,
            'percentages': percentages,
            'descriptions': descriptions,
            'chart_data': json.dumps(scores)
        }
        
        return render(request, 'predictor/result.html', context)
        
    except PersonalityPrediction.DoesNotExist:
        return render(request, 'predictor/home.html', {
            'error': 'Prediction not found.'
        })

def about(request):
    """About page with project information"""
    return render(request, 'predictor/about.html')

def generate_descriptions(scores):
    """Generate personality descriptions based on scores"""
    descriptions = {}
    
    trait_descriptions = {
        'Openness': {
            'high': 'Creative, curious, and imaginative. You enjoy new experiences and abstract thinking.',
            'medium': 'Balanced between traditional and novel experiences.',
            'low': 'Practical, conventional, and prefers routine over new experiences.'
        },
        'Conscientiousness': {
            'high': 'Organized, disciplined, and goal-oriented. You are reliable and self-controlled.',
            'medium': 'Moderately organized with balanced planning.',
            'low': 'Flexible, spontaneous, and less concerned with deadlines.'
        },
        'Extraversion': {
            'high': 'Outgoing, energetic, and sociable. You enjoy being around people.',
            'medium': 'Balanced between social time and solitude.',
            'low': 'Reserved, thoughtful, and prefers smaller groups or solitude.'
        },
        'Agreeableness': {
            'high': 'Cooperative, trusting, and compassionate. You consider others\' feelings.',
            'medium': 'Balanced between cooperation and self-interest.',
            'low': 'Critical, competitive, and more focused on own needs.'
        },
        'Neuroticism': {
            'high': 'Sensitive and emotionally responsive. You may experience stress more intensely.',
            'medium': 'Emotionally balanced with moderate stress response.',
            'low': 'Calm, secure, and emotionally stable.'
        }
    }
    
    for trait, score in scores.items():
        if score >= 0.7:
            level = 'high'
        elif score >= 0.4:
            level = 'medium'
        else:
            level = 'low'
        
        descriptions[trait] = trait_descriptions[trait][level]
    
    return descriptions
