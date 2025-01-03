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
        harvest_result = req.POST.get('harvest_result')
        result = req.POST.get('result')
        netto = req.POST.get('netto')
        harvest_power = req.POST.get('power')

        data = pd.DataFrame({
            'Hasil Panen': [float(harvest_result)],
            'tenaga panen': [float(harvest_power)],
            'Hasil': [float(result)],
            'Netto': [float(netto)],
            'Tonase': [1]
        })

        columns_to_scale = ['Hasil Panen', 'tenaga panen', 'Hasil', 'Netto', 'Tonase']
        scaled_data = scaler.transform(data[columns_to_scale])
        data[columns_to_scale] = scaled_data
        data = data.drop('Tonase', axis=1)

        pred = model.predict(data)
        pred_result = get_pred(pred, scaler)
        pred = int(np.round(pred_result))

        context = {
            'tonase': pred,
            'harvest_result': harvest_result,
            "harvest_power": harvest_power,
            "result": result,
            "netto": netto
        }

        return render(req, get_template('main'), context=context)

    return render(req, get_template('main'))


def save_pred(request):
    if request.method == "POST":
        harvest_result = request.POST.get("harvest_result")
        harvest_power = request.POST.get("harvest_power")
        result = request.POST.get("result")
        netto = request.POST.get("netto")
        tonase_result = request.POST.get("tonase_result")
        res_model = Result.objects.create(
            harvest_result=harvest_result,
            harvest_power=harvest_power,
            result=result,
            netto=netto,
            tonase=tonase_result
        )
        res_model.save()
        return redirect('prediction')
    return Http404()
