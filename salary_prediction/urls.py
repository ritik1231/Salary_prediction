from django.urls import path

from . import views

urlpatterns = [
    path("" , views.job , name = "job"),
    path("predict",views.predict , name="predict")
]

