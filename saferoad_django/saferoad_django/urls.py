from django.contrib import admin
from django.urls import path
from core import views  # Importamos nuestra vista

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'), # Esto hace que cargue en la página principal
]