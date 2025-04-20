"""
Esta clase fue creada para envolver un modelo LSTM de PyTorch y hacerlo compatible con el ecosistema de scikit-learn.
Esto permite integrar el modelo fácilmente en flujos de trabajo que utilizan herramientas como `GridSearchCV`, `Pipeline`
y funciones de evaluación que esperan métodos estándar como `fit()`, `predict()`, `get_params()` y `set_params()`.
Además, facilita la experimentación y reproducción de modelos al permitir ajustar hiperparámetros y reconstruir
el modelo de forma automática, manteniendo la flexibilidad y el control que ofrece PyTorch.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])


class SklearnLikeLSTM(BaseEstimator, RegressorMixin):
    def __init__(self, input_shape=None, hidden_size=50, epochs=10, batch_size=32, lr=0.001, device=None):
        self.input_shape = input_shape
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None  # Se crea en fit

    def _build_model(self):
        input_size = self.input_shape[-1]
        model = LSTMModel(input_size=input_size, hidden_size=self.hidden_size)
        return model.to(self.device)

    def fit(self, X, y):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        self.input_shape = X.shape[1:] if self.input_shape is None else self.input_shape
        self.model = self._build_model()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for _ in range(self.epochs):
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                output = self.model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X.to(self.device))
        return preds.cpu().numpy().squeeze()

    def get_params(self, deep=True):
        return {
            "input_shape": self.input_shape,
            "hidden_size": self.hidden_size,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "device": self.device,
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self