import re
import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.linear_model import LinearRegression
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Optional Twitter integration
try:
    import tweepy
    TWITTER_AVAILABLE = True
except ImportError:
    TWITTER_AVAILABLE = False
    print("Warning: tweepy not installed. Twitter integration will be disabled.")

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class PersonalityPredictor:
    def __init__(self):
        """Initialize the personality prediction model"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load BERT model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        
        # Initialize personality models (will be loaded or trained)
        self.personality_models = {}
        self.load_models()
        
        # Load Twitter API credentials from Django settings
        try:
            from django.conf import settings
            self.twitter_api_key = getattr(settings, 'TWITTER_API_KEY', '')
            self.twitter_api_secret = getattr(settings, 'TWITTER_API_SECRET_KEY', '')
            self.twitter_access_token = getattr(settings, 'TWITTER_ACCESS_TOKEN', '')
            self.twitter_access_token_secret = getattr(settings, 'TWITTER_ACCESS_TOKEN_SECRET', '')
        except:
            self.twitter_api_key = ''
            self.twitter_api_secret = ''
            self.twitter_access_token = ''
            self.twitter_access_token_secret = ''
    
    def clean_text(self, text):
        """Clean and preprocess text data"""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags (keep the text)
        text = re.sub(r'@\w+|#', '', text)
        
        # Remove emojis and special characters
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        filtered_text = [word for word in word_tokens if word not in stop_words]
        
        return ' '.join(filtered_text)
    
    def get_bert_embedding(self, text):
        """Convert text to BERT embeddings"""
        # Clean the text first
        cleaned_text = self.clean_text(text)
        
        # Tokenize and encode
        inputs = self.tokenizer(
            cleaned_text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)
        
        # Get BERT output
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            
        # Use the [CLS] token embedding or mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        return embeddings.flatten()
    
    def load_models(self):
        """Load pre-trained personality models"""
        traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        
        for trait in traits:
            try:
                # Try to load pre-trained model
                self.personality_models[trait] = joblib.load(f'models/{trait}_model.pkl')
                print(f"Loaded {trait} model")
            except FileNotFoundError:
                # If no pre-trained model, create a new one
                self.personality_models[trait] = LinearRegression()
                print(f"Created new {trait} model")
    
    def train_models(self, training_data_path):
        """Train personality prediction models"""
        # Load training data
        df = pd.read_csv(training_data_path)
        
        # Prepare features (BERT embeddings) and targets (personality scores)
        X = []
        y = {trait: [] for trait in self.personality_models.keys()}
        
        for _, row in df.iterrows():
            # Get BERT embedding for text
            embedding = self.get_bert_embedding(row['text'])
            X.append(embedding)
            
            # Add personality scores
            for trait in self.personality_models.keys():
                y[trait].append(row[trait])
        
        X = np.array(X)
        
        # Train models for each trait
        for trait in self.personality_models.keys():
            self.personality_models[trait].fit(X, y[trait])
            
            # Save the model
            joblib.dump(self.personality_models[trait], f'models/{trait}_model.pkl')
            print(f"Trained and saved {trait} model")
    
    def predict_personality(self, text):
        """Predict personality traits from text"""
        # Get BERT embedding
        embedding = self.get_bert_embedding(text).reshape(1, -1)
        
        # Predict scores for each trait
        scores = {}
        for trait, model in self.personality_models.items():
            score = model.predict(embedding)[0]
            # Ensure score is between 0 and 1
            scores[trait] = max(0, min(1, score))
        
        return scores
    
    def fetch_twitter_posts(self, username, count=50):
        """Fetch recent tweets from a Twitter user"""
        if not TWITTER_AVAILABLE:
            raise ValueError("Twitter integration is not available. Please install tweepy: pip install tweepy")
        
        try:
            # Use Twitter API v2 with Bearer Token authentication
            import requests
            
            # Get bearer token from settings
            try:
                from django.conf import settings
                bearer_token = getattr(settings, 'TWITTER_BEARER_TOKEN', '')
            except:
                bearer_token = ''
            
            # Clean the bearer token (decode URL encoding if needed)
            if bearer_token:
                import urllib.parse
                bearer_token = urllib.parse.unquote(bearer_token)
            
            if not bearer_token:
                raise ValueError("Twitter Bearer Token not configured")
            
            # Twitter API v2 endpoint for user tweets
            headers = {
                "Authorization": f"Bearer {bearer_token}",
                "Content-Type": "application/json"
            }
            
            # First get user ID from username
            user_url = f"https://api.twitter.com/2/users/by/username/{username}"
            
            print(f"Fetching user data for: {username}")
            user_response = requests.get(user_url, headers=headers)
            
            print(f"User API response status: {user_response.status_code}")
            if user_response.status_code == 402:
                # Credits depleted - provide demo data
                return self.get_demo_twitter_data(username)
            elif user_response.status_code != 200:
                print(f"User API response: {user_response.text}")
                raise Exception(f"Could not find user: {username}. Status: {user_response.status_code}")
            
            user_data = user_response.json()
            user_id = user_data['data']['id']
            print(f"Found user ID: {user_id}")
            
            # Get user's tweets
            tweets_url = f"https://api.twitter.com/2/users/{user_id}/tweets"
            params = {
                "max_results": min(count, 100),  # API limit is 100
                "tweet.fields": "created_at,public_metrics"
            }
            
            print(f"Fetching tweets for user ID: {user_id}")
            tweets_response = requests.get(tweets_url, headers=headers, params=params)
            
            print(f"Tweets API response status: {tweets_response.status_code}")
            if tweets_response.status_code == 402:
                # Credits depleted - provide demo data
                return self.get_demo_twitter_data(username)
            elif tweets_response.status_code != 200:
                print(f"Tweets API response: {tweets_response.text}")
                raise Exception(f"Could not fetch tweets: {tweets_response.text}")
            
            tweets_data = tweets_response.json()
            
            # Combine tweet text
            if 'data' in tweets_data:
                tweets = tweets_data['data']
                combined_text = ' '.join([tweet['text'] for tweet in tweets])
                print(f"Successfully fetched {len(tweets)} tweets")
                return combined_text
            else:
                return "No tweets found for this user."
            
        except Exception as e:
            raise Exception(f"Error fetching Twitter data: {str(e)}")
    
    def get_demo_twitter_data(self, username):
        """Provide demo Twitter data when API credits are depleted"""
        demo_tweets = {
            'elonmusk': "Working on making life multiplanetary. SpaceX will make humanity multiplanetary. Tesla accelerating sustainable energy. Neuralink helping brain-computer interfaces. The Boring Company solving urban traffic. Twitter/X protecting free speech. Innovation is key to humanity's future.",
            'nasa': "Exploring the universe for the benefit of all. Mars rover sending amazing images. James Webb telescope discovering distant galaxies. International Space Station advancing science. Artemis mission returning to the Moon. Space exploration inspires future generations.",
            'natgeo': "Protecting our planet through storytelling. Wildlife photography capturing nature's beauty. Documentaries exploring diverse cultures. Environmental conservation is crucial. Climate change affects us all. Science education matters for our future.",
            'default': f"Welcome to {username}'s Twitter profile! This is demo data since API credits are depleted. In a real implementation, this would contain actual tweets from @{username}. The system would analyze their personality based on their social media posts, communication style, and engagement patterns."
        }
        
        return demo_tweets.get(username.lower(), demo_tweets['default'])
    
    def set_twitter_credentials(self, api_key, api_secret, access_token, access_token_secret):
        """Set Twitter API credentials"""
        self.twitter_api_key = api_key
        self.twitter_api_secret = api_secret
        self.twitter_access_token = access_token
        self.twitter_access_token_secret = access_token_secret

# Create a sample training data generator for demonstration
def create_sample_training_data():
    """Create sample training data for demonstration purposes"""
    sample_data = {
        'text': [
            "I love exploring new ideas and trying different things every day.",
            "I always plan my schedule carefully and stick to my goals.",
            "I enjoy being around people and going to social events.",
            "I always try to help others and maintain harmony in relationships.",
            "I often feel worried and stressed about small things."
        ],
        'openness': [0.9, 0.3, 0.6, 0.5, 0.4],
        'conscientiousness': [0.4, 0.9, 0.5, 0.7, 0.3],
        'extraversion': [0.6, 0.3, 0.9, 0.7, 0.2],
        'agreeableness': [0.7, 0.6, 0.8, 0.9, 0.4],
        'neuroticism': [0.3, 0.2, 0.4, 0.3, 0.9]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('training_data/personality_data.csv', index=False)
    print("Sample training data created at training_data/personality_data.csv")
