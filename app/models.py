from django.db import models


class Result(models.Model):
    date = models.DateField(auto_now_add=True)
    harvest_power = models.FloatField(null=False, blank=False)
    result = models.FloatField(null=False, blank=False)
    netto = models.FloatField(null=False, blank=False)
    tonnage = models.FloatField(null=False, blank=False)
    harvest_result = models.FloatField(null=False, blank=False)

    def __str__(self):
        return "prediksi tanggal {} mendapatkan hasil panen : {}".format(self.date, self.harvest_result)
