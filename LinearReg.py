import numpy as np


class LinearRegression:

    def __init__(self, mae_metric=False):
        """
            @param mae_metrics: В случае True необходимо следить за
            метрикой MAE во время обучения, иначе - за метрикой MSE
        """
        self.metric = self.calc_mse_metric if not mae_metric else self.calc_mae_metric

    def calc_mae_metric(self, preds, y):
        """
            @param preds: предсказания модели
            @param y: истиные значения
            @return mae: значение MAE
        """
        return sum(sum(abs(preds - y))) / (y.shape[0])

    def calc_mse_metric(self, preds, y):
        """
            @param preds: предсказания модели
            @param y: истиные значения
            @return mse: значение MSE
        """
        return sum(sum((preds - y) ** 2)) / (y.shape[0])

    def init_weights(self, input_size, output_size):
        """
            Инициализирует параметры модели
            W - матрица размерности (input_size, output_size)
            инициализируется рандомными числами из
            нормального распределения со средним 0 и стандартным отклонением 0.01
            b - вектор размерности (1, output_size)
            инициализируется нулями
        """
        np.random.seed(42)
        self.W = np.random.normal(loc=0.0, scale=0.01, size=(input_size, output_size))
        self.b = np.zeros((output_size))
        self.W_b = np.vstack((self.W, self.b))

    def fit(self, X, y, num_epochs=1000, lr=0.001):
        """
            Обучение модели линейной регрессии методом градиентного спуска
            @param X: размерности (num_samples, input_shape)
            @param y: размерности (num_samples, output_shape)
            @param num_epochs: количество итераций градиентного спуска
            @param lr: шаг градиентного спуска
            @return metrics: вектор значений метрики на каждом шаге градиентного
            спуска. В случае mae_metric==True вычисляется метрика MAE
            иначе MSE
        """
        self.init_weights(X.shape[1], y.shape[1])
        metrics = []
        for _ in range(num_epochs):
            preds = self.predict(X)
            W_b_grad = 2 * (np.hstack((X, np.ones((len(X), 1)))).T @ (
                        (np.hstack((X, np.ones((len(X), 1)))) @ self.W_b) - y)) \
                       / (y.shape[0])
            self.W_b -= lr * W_b_grad
            metrics.append(self.metric(preds, y))
        return metrics

    def predict(self, X):
        """
            Думаю, тут все понятно. Сделать свои предсказания :)
        """
        return np.hstack((X, np.ones((len(X), 1)))) @ self.W_b
