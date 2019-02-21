"""myapp URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.8/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import include, url
from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static
from . import views



urlpatterns = [
    url(r'^admin/', include(admin.site.urls)),
    url(r'^$', views.home, name='home'),
    url(r'preloader', views.preloader, name='preloader'),
    url(r'pre1', views.pre1, name='pre1'),
    url(r'loader2', views.loader2, name='loader2'),
    url(r'preload3', views.preload3, name='preload3'),
    url(r'startTraining', views.startTraining, name='startTraining'),
    url(r'startAnalysing', views.startAnalysing, name='startAnalysing'),
    url(r'startLearning', views.startLearning, name='startLearning'),
    url(r'startSampling', views.startSampling, name='startSampling'),

]


if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
