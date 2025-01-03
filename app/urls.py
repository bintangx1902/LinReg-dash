from django.urls import path
from .views import *
urlpatterns = [
    path("", prediction, name='prediction'),
    path('save', save_pred, name='save')
]