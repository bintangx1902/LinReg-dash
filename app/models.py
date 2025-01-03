from django.db import models


class Result(models.Model):
    date = models.DateField(auto_now_add=True)
    harvest_result = models.IntegerField(null=False, blank=False)
    harvest_power = models.IntegerField(null=False, blank=False)
    result = models.IntegerField(null=False, blank=False)
    netto = models.IntegerField(null=False, blank=False)
    tonase = models.IntegerField(null=False, blank=False)

    def __str__(self):
        return "prediksi tanggal {} dengan hasil tonase : {}".format(self.date, self.tonase)
