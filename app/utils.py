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

        X = df[['Tanggal', 'Hasil', 'tenaga panen']]
        y = df['Hasil Panen']

        X['Tanggal'] = pd.to_datetime(X['Tanggal'])

        df_temp = X.copy()
        df_temp['Harvest Result'] = y
        df_temp['Bulan'] = df_temp['Tanggal'].dt.to_period('M').astype(str)
        df_monthly = df_temp.groupby('Bulan')['Harvest Result'].agg(['min', 'mean', 'max']).reset_index()

        x_train, x_test, y_train, y_test = train_test_split(X.drop('Tanggal', axis=1), y, test_size=0.2,
                                                            shuffle=False)
        train_date, test_date = train_test_split(X['Tanggal'], test_size=0.2, shuffle=False)

        pred = model.predict(x_test)
        train_pred = model.predict(x_train)

        pred_series = pd.Series(pred, index=x_test.index)
        train_pred_series = pd.Series(train_pred, index=x_train.index)

        full_prediction = pd.concat([train_pred_series, pred_series], axis=0, ignore_index=True)

        df_full = pd.DataFrame({
            'Tanggal': X['Tanggal'],
            'Actual': y,
            'Prediction': full_prediction
        })

        df_full['Bulan'] = df_full['Tanggal'].dt.to_period('M').astype(str)

        df_full_monthly = df_full.groupby('Bulan').agg({
            'Actual': 'max',
            'Prediction': 'max'
        }).reset_index()

        df_power_agg = X.copy()
        df_power_agg['Harvest Result'] = y
        df_power_agg['Bulan'] = df_power_agg['Tanggal'].dt.to_period('M').astype(str)

        # Agregasi Max untuk kedua kolom
        df_monthly_power = df_power_agg.groupby('Bulan').agg({
            'Harvest Result': 'max',
            'tenaga panen': 'max'
        }).reset_index()

        harvest_result_fig = go.Figure()
        # harvest_result_fig.add_trace(go.Scatter(x=X['Tanggal'], y=y, name='Harvest Result'))
        #
        # harvest_result_fig.update_layout(
        #     title='Sebaran Hasil Panen',
        #     xaxis_tickangle=-45,
        #     xaxis_title='Tanggal',
        #     yaxis_title='Harvest Result',
        # )

        # Trace untuk min
        harvest_result_fig.add_trace(go.Bar(
            x=df_monthly['Bulan'],
            y=df_monthly['min'],
            name='Min',
            marker_color='indianred'
        ))

        # Trace untuk Average
        harvest_result_fig.add_trace(go.Bar(
            x=df_monthly['Bulan'],
            y=df_monthly['mean'],
            name='Average',
            marker_color='lightsalmon'
        ))

        # Trace untuk Maximum
        harvest_result_fig.add_trace(go.Bar(
            x=df_monthly['Bulan'],
            y=df_monthly['max'],
            name='Max',
            marker_color='seagreen'
        ))

        # 6. Update Layout
        harvest_result_fig.update_layout(
            title='Agregat Hasil Panen per Bulan (Min, Avg, Max)',
            xaxis_title='Bulan',
            yaxis_title='Harvest Result',
            barmode='group',  # Membuat bar bersampingan
            xaxis_tickangle=-45
        )

        power_and_result = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Max Hasil Panen (HA) per Bulan", "Max Tenaga Panen per Bulan")
        )

        # Subplot 1: Max Hasil Panen
        power_and_result.add_trace(
            go.Bar(
                x=df_monthly_power['Bulan'],
                y=df_monthly_power['Harvest Result'],
                name='Max Hasil Panen',
                marker_color='teal'
            ),
            row=1, col=1
        )

        # Subplot 2: Max Tenaga Panen
        power_and_result.add_trace(
            go.Bar(
                x=df_monthly_power['Bulan'],
                y=df_monthly_power['tenaga panen'],
                name='Max Tenaga Panen',
                marker_color='goldenrod'
            ),
            row=1, col=2
        )

        # 3. Update Axes dan Layout
        power_and_result.update_xaxes(title_text="Bulan", tickangle=-45, row=1, col=1)
        power_and_result.update_yaxes(title_text="Hasil Panen", row=1, col=1)

        power_and_result.update_xaxes(title_text="Bulan", tickangle=-45, row=1, col=2)
        power_and_result.update_yaxes(title_text="Jumlah Tenaga", row=1, col=2)

        power_and_result.update_layout(
            title_text="Analisis Maksimum Bulanan: Hasil vs Tenaga",
            showlegend=True,
            height=450,
            template='plotly_white'
        )

        comparison = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Grafik Hasil Panen dengan Hasil (HA)", "Grafik Hasil Panen dengan Tenaga Panen")
        )

        comparison.add_trace(
            go.Scatter(
                x=y,
                y=X['Hasil'],
                name='Hasil (HA)',
                mode='markers'
            ),
            row=1, col=1
        )

        comparison.add_trace(
            go.Scatter(
                x=y,
                y=X['tenaga panen'],
                name='Tenaga Panen',
                mode='markers'
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
        )

        test_fig = go.Figure()
        test_fig.add_trace(go.Scatter(x=test_date, y=y_test, mode='lines', name='Data Asli'))
        test_fig.add_trace(go.Scatter(x=test_date, y=pred_series, mode='lines', name='Prediksi'))

        test_fig.update_layout(
            title="Perbandingan Data Asli vs Prediksi",
            xaxis_title="Tanggal",
            yaxis_title="Hasil Panen"
        )

        full_fig = go.Figure()
        full_fig.add_trace(go.Scatter(
            x=X['Tanggal'],
            y=y,
            mode='lines',
            name='Data Asli'
        ))
        full_fig.add_trace(go.Scatter(
            x=X['Tanggal'],
            y=full_prediction,
            mode='lines',
            name='Prediksi'
        ))

        full_fig.update_layout(
            title="Perbandingan Data Asli vs Prediksi pada Keseluruhan Data",
            xaxis_title="Tanggal",
            yaxis_title="Hasil Panen",
            template="plotly_white"
        )

        full_fig2 = go.Figure()
        full_fig2.add_trace(go.Bar(
            x=df_full_monthly['Bulan'],
            y=df_full_monthly['Actual'],
            name='Data Asli (Max)',
            marker_color='royalblue'
        ))

        full_fig2.add_trace(go.Bar(
            x=df_full_monthly['Bulan'],
            y=df_full_monthly['Prediction'],
            name='Prediksi (Max)',
            marker_color='orange'
        ))

        full_fig2.update_layout(
            title="Perbandingan Max Data Asli vs Prediksi per Bulan",
            xaxis_title="Bulan",
            yaxis_title="Hasil Panen",
            barmode='group',
            xaxis_tickangle=-45,
            template="plotly_white"
        )
        extra_context['harvest_result_fig'] = harvest_result_fig.to_json()
        extra_context['power_and_result'] = power_and_result.to_json()
        extra_context['comparison'] = comparison.to_json()
        extra_context["test_fig"] = test_fig.to_json()
        extra_context['full_fig'] = full_fig.to_json()
        extra_context['full_fig2'] = full_fig2.to_json()

        return super().changelist_view(request, extra_context=extra_context)
