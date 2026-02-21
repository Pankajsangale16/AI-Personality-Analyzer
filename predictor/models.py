from django.db import models

class PersonalityPrediction(models.Model):
    text_input = models.TextField()
    twitter_handle = models.CharField(max_length=100, blank=True, null=True)
    openness_score = models.FloatField()
    conscientiousness_score = models.FloatField()
    extraversion_score = models.FloatField()
    agreeableness_score = models.FloatField()
    neuroticism_score = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Prediction for {self.twitter_handle or 'Manual Input'} - {self.created_at}"
