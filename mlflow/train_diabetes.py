import warnings
import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import enet_path
import mlflow
import mlflow.sklearn

# Helper function from the notebook

def plot_enet_descent_path(X, y, l1_ratio):
    eps = 5e-3  # the smaller it is the longer is the path
    global image
    print("Computing regularization path using ElasticNet.")
    alphas_enet, coefs_enet, _ = enet_path(X, y, eps=eps, l1_ratio=l1_ratio)
    fig = plt.figure(1)
    ax = plt.gca()
    colors = cycle(['b', 'r', 'g', 'c', 'k'])
    neg_log_alphas_enet = -np.log10(alphas_enet)
    for coef_e, c in zip(coefs_enet, colors):
        plt.plot(neg_log_alphas_enet, coef_e, linestyle='--', c=c)
    plt.xlabel('-Log(alpha)')
    plt.ylabel('coefficients')
    title = 'ElasticNet Path by alpha for l1_ratio = ' + str(l1_ratio)
    plt.title(title)
    plt.axis('tight')
    image = fig
    fig.savefig("ElasticNet-paths.png")
    plt.close(fig)
    return image


def train_diabetes(data, in_alpha, in_l1_ratio):
    """Train an ElasticNet model on the given diabetes data."""

    def eval_metrics(actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    warnings.filterwarnings("ignore")
    np.random.seed(40)

    train, test = train_test_split(data)
    train_x = train.drop(["progression"], axis=1)
    test_x = test.drop(["progression"], axis=1)
    train_y = train[["progression"]]
    test_y = test[["progression"]]

    alpha = float(in_alpha) if float(in_alpha) is not None else 0.05
    l1_ratio = float(in_l1_ratio) if float(in_l1_ratio) is not None else 0.05

    mlflow.sklearn.autolog()
    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)
        predicted_qualities = lr.predict(test_x)
        rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)
        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.sklearn.log_model(lr, "model")
        import tempfile, os
        model_dir = tempfile.mkdtemp()
        modelpath = os.path.join(model_dir, "model-%f-%f" % (alpha, l1_ratio))
        mlflow.sklearn.save_model(lr, modelpath)
        image = plot_enet_descent_path(data.drop('progression', axis=1).values, data['progression'].values, l1_ratio)
        mlflow.log_artifact("ElasticNet-paths.png")
    return rmse, mae, r2
