import numpy as np
import pandas as pd
from sklearn import datasets
from pathlib import Path
import sys

# add module path
MODULE_DIR = Path(__file__).resolve().parents[1] / "mlflow"
sys.path.insert(0, str(MODULE_DIR))
from train_diabetes import train_diabetes


def test_train_diabetes():
    diabetes = datasets.load_diabetes()
    X = diabetes.data
    y = diabetes.target
    Y = np.array([y]).transpose()
    d = np.concatenate((X, Y), axis=1)
    cols = [
        'age', 'sex', 'bmi', 'bp', 's1', 's2',
        's3', 's4', 's5', 's6', 'progression'
    ]
    data = pd.DataFrame(d, columns=cols)
    rmse, mae, r2 = train_diabetes(data, 0.01, 0.01)
    assert isinstance(rmse, float)
    assert isinstance(mae, float)
    assert isinstance(r2, float)
