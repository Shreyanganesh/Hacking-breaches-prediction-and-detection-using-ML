from django.db import models

# Create your models here.
class Cyber_Breaches(models.Model):
    Organisation=models.CharField(max_length=500)
    Records=models.CharField(max_length=500)
    Information_Source=models.CharField(max_length=500)
    Source_URL=models.CharField(max_length=500)
    Latitude=models.CharField(max_length=500)
    Longitude=models.CharField(max_length=500)
    Type_Of_Breaches=models.CharField(max_length=500)