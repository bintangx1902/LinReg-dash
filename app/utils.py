import json
import os

import joblib
import numpy as np
import pandas as pd
from django.conf import settings
from django.contrib import admin
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split


def load_scaler():
    return joblib.load(os.path.join(settings.MEDIA_ROOT, 'model', 'scaler.pkl'))


def load_model():
    path = os.path.join(settings.MEDIA_ROOT, 'model', 'final_model.pkl')
    return joblib.load(path)


def get_pred(prediction, scaler):
    features_filled = np.ones((len(prediction), 4))
    tonase_column = prediction.reshape(-1, 1)
    combined = np.hstack((tonase_column, features_filled))

    final_unscaled = scaler.inverse_transform(combined)
    final_df_pred = pd.DataFrame(final_unscaled,
                                 columns=['Hasil Panen', 'tenaga panen', 'Hasil', 'Netto', 'Tonase'])
    return final_df_pred['Hasil Panen'].to_numpy()[0]


class ModelAnalyticsAdminView(admin.ModelAdmin):
    change_list_template = "admin/analytics.html"

    def changelist_view(self, request, extra_context={}):
        model = load_model()

        path = os.path.join(settings.MEDIA_ROOT, 'model', 'ScaledData.xlsx')
        df = pd.read_excel(path).dropna()

        X = df[['Hasil', 'tenaga panen']]
        y = df['Hasil Panen']

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        pred = model.predict(x_test)
        pred_series = pd.Series(pred, index=x_test.index)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_test.index, y=y_test, mode='lines', name='Data Asli'))
        fig.add_trace(go.Scatter(x=pred_series.index, y=pred_series, mode='lines', name='Prediksi'))

        fig.update_layout(
            title="Perbandingan Data Asli vs Prediksi",
            xaxis_title="Index",
            yaxis_title="Hasil Panen"
        )

        extra_context["fig"] = fig.to_json()

        return super().changelist_view(request, extra_context=extra_context)


