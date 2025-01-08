from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('predictionapp.urls')),  # Include URLs from predictionapp
]
