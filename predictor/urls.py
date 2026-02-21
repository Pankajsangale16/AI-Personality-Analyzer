from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('predict/', views.predict_personality, name='predict_personality'),
    path('result/<int:prediction_id>/', views.result, name='result'),
    path('about/', views.about, name='about'),
]
