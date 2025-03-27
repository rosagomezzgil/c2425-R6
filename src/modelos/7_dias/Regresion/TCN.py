import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import optuna
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from torch.nn.utils import weight_norm

# Definir el modelo TCN
class TemporalConvNet(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            layers.append(weight_norm(nn.Conv1d(input_size if i == 0 else num_channels[i - 1],
                                                 num_channels[i], kernel_size,
                                                 padding=padding, dilation=dilation)))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        x = self.network(x)
        x = x[:, :, -1]  # Último valor de la secuencia
        return self.fc(x)

# Cargar los datos
def cargar_datos(ruta_parquet):
    df = pd.read_parquet(ruta_parquet)
    return df

# Preprocesar datos
def preprocesar_datos(df):
    df = df.drop(columns=['tendencia_futura_7d', 'pct_cambio_30d', 'tendencia_futura_30d'], errors='ignore')
    X = df.drop(columns=['pct_cambio_7d'])
    y = df['pct_cambio_7d']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y.values, scaler

# Dividir los datos
def dividir_datos_temporalmente(X, y, train_size=0.8, val_size=0.1):
    n = len(X)
    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    return X_train, X_val, X_test, y_train, y_val, y_test

def objective(trial, X_train, y_train, X_val, y_val):
    num_channels = [trial.suggest_int(f'num_channels_{i}', 8, 64) for i in range(3)]
    kernel_size = trial.suggest_int('kernel_size', 2, 5)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)

    model = TemporalConvNet(X_train.shape[1], num_channels, kernel_size, dropout)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(2)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(2)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model.train()
    for epoch in range(10):  # Puedes ajustar el número de épocas para Optuna
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X).squeeze()
            loss = criterion(output, batch_y.squeeze())
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_tensor).squeeze().numpy()
        val_true = y_val_tensor.squeeze().numpy()

    mse = mean_squared_error(val_true, val_preds)
    return mse  # Optuna buscará minimizar este valor

# Función para encontrar los mejores hiperparámetros
def buscar_mejores_hiperparametros(X_train, y_train, X_val, y_val):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=20)
    print("Mejores hiperparámetros:", study.best_params)
    return study.best_params

def entrenar_y_evaluar(X_train, X_val, X_test, y_train, y_val, y_test, mejores_params):
    model = TemporalConvNet(X_train.shape[1],
                             [mejores_params[f'num_channels_{i}'] for i in range(3)],
                             mejores_params['kernel_size'],
                             mejores_params['dropout'])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=mejores_params['lr'])

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(2)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(2)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(2)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model.train()
    for epoch in range(50):  # Se puede ajustar el número de épocas
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X).squeeze()
            loss = criterion(output, batch_y.squeeze())
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_tensor).squeeze().numpy()
        test_true = y_test_tensor.squeeze().numpy()

    mse = mean_squared_error(test_true, test_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_true, test_preds)
    r2 = r2_score(test_true, test_preds)

    print(f"MSE en test: {mse:.4f}")
    print(f"RMSE en test: {rmse:.4f}")
    print(f"MAE en test: {mae:.4f}")
    print(f"R2 en test: {r2:.4f}")

# Función principal
def main():
    ruta_parquet = r"../../../../data/parquet/data_7d.parquet"
    df = cargar_datos(ruta_parquet)
    X, y, scaler = preprocesar_datos(df)
    X_train, X_val, X_test, y_train, y_val, y_test = dividir_datos_temporalmente(X, y)
    mejores_params = buscar_mejores_hiperparametros(X_train, y_train, X_val, y_val)
    print("Hiperparámetros óptimos obtenidos:", mejores_params)
    entrenar_y_evaluar(X_train, X_val, X_test, y_train, y_val, y_test, mejores_params)

if __name__ == "__main__":
    main()
