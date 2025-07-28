import numpy as np
import pandas as pd
from django.http import Http404
from django.shortcuts import render, redirect

from .models import Result
from .utils import load_model, load_scaler, get_pred

# Load the model and scaler
model = load_model()
scaler = load_scaler()


def get_template(file_name):
    return 'app/{}.html'.format(file_name)


def prediction(req):
    if req.method == 'POST':
        result = req.POST.get('result')
        netto = req.POST.get('netto')
        harvest_power = req.POST.get('power')
        tonnage = req.POST.get('tonnage')

        data = pd.DataFrame({
            'Hasil Panen': [1.],
            'tenaga panen': [float(harvest_power)],
            'Hasil': [float(result)],
            'Netto': [float(netto)],
            'Tonase': [float(tonnage)],
        })

        columns_to_scale = ['Hasil Panen', 'tenaga panen', 'Hasil', 'Netto', 'Tonase']
        scaled_data = scaler.transform(data[columns_to_scale])
        data[columns_to_scale] = scaled_data
        data = data.drop('Hasil Panen', axis=1)

        pred = model.predict(data)
        pred_result = get_pred(pred, scaler)
        pred = int(np.round(pred_result))

        context = {
            'tonnage': tonnage,
            "harvest_power": harvest_power,
            "result": result,
            "netto": netto,
            'harvest_result': pred
        }

        return render(req, get_template('main'), context=context)

    return render(req, get_template('main'))


def save_pred(request):
    if request.method == "POST":
        harvest_result = request.POST.get("harvest_result")
        harvest_power = request.POST.get("harvest_power")
        result = request.POST.get("result")
        netto = request.POST.get("netto")
        tonnage = request.POST.get("tonnage")
        res_model = Result.objects.create(
            harvest_result=harvest_result,
            harvest_power=harvest_power,
            result=result,
            netto=netto,
            tonnage=tonnage,
        )
        res_model.save()
        return redirect('prediction')
    return Http404()
