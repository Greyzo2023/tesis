from django.urls import path
from django.urls import path
from .views import FileUploadView, ResultView 

urlpatterns = [
    path('upload/', FileUploadView.as_view(), name='file-upload'),
    path('resultado/', ResultView.as_view(), name='resultado'),  
]