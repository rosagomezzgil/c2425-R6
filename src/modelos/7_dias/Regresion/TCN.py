import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import optuna
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
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
def dividir_datos(X, y, test_size=0.2, val_size=0.2, random_state=42):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size + val_size, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size / (test_size + val_size), random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test

# Búsqueda de hiperparámetros con Optuna
def objective(trial, X_train, y_train, X_val, y_val):
    num_channels = [trial.suggest_int(f'num_channels_{i}', 16, 128) for i in range(3)]
    kernel_size = trial.suggest_int('kernel_size', 2, 5)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)

    model = TemporalConvNet(X_train.shape[1], num_channels, kernel_size, dropout)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model.train()
    for epoch in range(10):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X).squeeze()
            loss = criterion(output, batch_y.squeeze())
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_tensor).squeeze()
        val_loss = criterion(val_preds, y_val_tensor.squeeze()).item()

    return val_loss

# Función para encontrar los mejores hiperparámetros
def buscar_mejores_hiperparametros(X_train, y_train, X_val, y_val):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=20)
    print("Mejores hiperparámetros:", study.best_params)
    return study.best_params

# Función principal
def main():
    ruta_parquet = "ruta/a/tu/dataset.parquet"
    df = cargar_datos(ruta_parquet)
    X, y, scaler = preprocesar_datos(df)
    X_train, X_val, X_test, y_train, y_val, y_test = dividir_datos(X, y)
    mejores_params = buscar_mejores_hiperparametros(X_train, y_train, X_val, y_val)
    print("Hiperparámetros óptimos obtenidos:", mejores_params)

if __name__ == "__main__":
    main()
