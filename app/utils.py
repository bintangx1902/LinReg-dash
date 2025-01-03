import os

import joblib
import numpy as np
import pandas as pd
from django.conf import settings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def load_scaler():
    return joblib.load(os.path.join(settings.MEDIA_ROOT, 'model', 'scaler.pkl'))


def load_model():
    path = os.path.join(settings.MEDIA_ROOT, 'model', 'linear_regression_model.pkl')
    return joblib.load(path)


def get_pred(prediction, scaler):
    features_filled = np.ones((len(prediction), 4))
    tonase_column = prediction.reshape(-1, 1)
    combined = np.hstack((features_filled, tonase_column))

    final_unscaled = scaler.inverse_transform(combined)
    final_df_pred = pd.DataFrame(final_unscaled,
                                 columns=['Hasil Panen', 'tenaga panen', 'Hasil', 'Netto', 'Tonase'])
    return final_df_pred['Tonase'].to_numpy()[0]
