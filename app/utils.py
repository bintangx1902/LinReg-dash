import os

import joblib
import numpy as np
import pandas as pd
from django.conf import settings
from django.contrib import admin
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

        harvest_result_fig = go.Figure()
        harvest_result_fig.add_trace(go.Scatter(x=X.index, y=y, name='Harvest Result'))

        harvest_result_fig.update_layout(
            title='Sebaran Hasil Panen',
            xaxis_tickangle=-45,
            xaxis_title='Index',
            yaxis_title='Harvest Result',
        )

        power_and_result = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Grafik sebaran data Hasil (HA)", "Grafik Sebaran Data Tenaga Panen")
        )

        power_and_result.add_trace(
            go.Scatter(
                x=X.index,
                y=y,
                name='Hasil (HA)',
                mode='lines'  # Gunakan 'lines' agar bentuknya garis seperti di gambar
            ),
            row=1, col=1
        )

        power_and_result.add_trace(
            go.Scatter(
                x=X.index,
                y=y,
                name='Tenaga Panen',
                mode='lines'
            ),
            row=1, col=2
        )

        power_and_result.update_xaxes(title_text="Tanggal", showgrid=True, row=1, col=1)
        power_and_result.update_yaxes(title_text="Hasil Panen", showgrid=True, row=1, col=1)

        power_and_result.update_xaxes(title_text="Tanggal", showgrid=True, row=1, col=2)
        power_and_result.update_yaxes(title_text="Tenaga Panen", showgrid=True, row=1, col=2)

        power_and_result.update_layout(
            title_text="Analisis Data Sawit",  # Judul Utama (Opsional)
            showlegend=True,
            height=400,  # Mengatur tinggi agar proporsional
            # width=1200 # Bisa diaktifkan jika ingin lebar fix
        )

        """ diff """

        comparison = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Grafik Hasil Panen dengan Hasil (HA)", "Grafik Hasil Panen dengan Tenaga Panen")
        )

        comparison.add_trace(
            go.Scatter(
                x=y,
                y=X['Hasil'],
                name='Hasil (HA)',
                mode='lines'  # Gunakan 'lines' agar bentuknya garis seperti di gambar
            ),
            row=1, col=1
        )

        comparison.add_trace(
            go.Scatter(
                x=y,
                y=X['tenaga panen'],
                name='Tenaga Panen',
                mode='lines'
            ),
            row=1, col=2
        )

        comparison.update_xaxes(title_text="Hasil Panen", showgrid=True, row=1, col=1)
        comparison.update_yaxes(title_text="Hasil (HA)", showgrid=True, row=1, col=1)

        comparison.update_xaxes(title_text="Hasil Panen", showgrid=True, row=1, col=2)
        comparison.update_yaxes(title_text="Tenaga Panen", showgrid=True, row=1, col=2)

        comparison.update_layout(
            title_text="Analisis Data Sawit",  # Judul Utama (Opsional)
            showlegend=True,
            height=400,  # Mengatur tinggi agar proporsional
            # width=1200 # Bisa diaktifkan jika ingin lebar fix
        )


        test_fig = go.Figure()
        test_fig.add_trace(go.Scatter(x=x_test.index, y=y_test, mode='lines', name='Data Asli'))
        test_fig.add_trace(go.Scatter(x=pred_series.index, y=pred_series, mode='lines', name='Prediksi'))

        test_fig.update_layout(
            title="Perbandingan Data Asli vs Prediksi",
            xaxis_title="Index",
            yaxis_title="Hasil Panen"
        )

        extra_context['harvest_result_fig'] = harvest_result_fig.to_json()
        extra_context['power_and_result'] = power_and_result.to_json()
        extra_context['comparison'] = comparison.to_json()
        extra_context["test_fig"] = test_fig.to_json()

        return super().changelist_view(request, extra_context=extra_context)
