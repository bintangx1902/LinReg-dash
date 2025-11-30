from django.contrib import admin
from app.models import Result
from django.db import models
from .utils import ModelAnalyticsAdminView


class AnalyticsModel(models.Model):
    class Meta:
        verbose_name = "Analytics Dashboard"
        verbose_name_plural = "Analytics Dashboard"


admin.site.register(Result)
admin.site.register(AnalyticsModel, ModelAnalyticsAdminView)
