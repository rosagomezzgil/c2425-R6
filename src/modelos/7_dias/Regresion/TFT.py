import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import RMSE
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting import Trainer

def load_and_preprocess_data(file_path):
    """
    Cargar y preprocesar el archivo parquet.
    """
    # Cargar el archivo Parquet
    df = pd.read_parquet(file_path)

    # Eliminar columnas no necesarias
    columns_to_drop = ['tendencia_futura_7d', 'pct_cambio_30d', 'tendencia_futura_30d']  # Columnas no necesarias
    df = df.drop(columns=columns_to_drop)

    # Escalar o normalizar las características
    scaler = StandardScaler()
    columns_to_scale = df.select_dtypes(include=['float64', 'int64']).columns
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    # Crear la variable respuesta pct_cambio_7d
    df['pct_cambio_7d'] = df['pct_cambio_7d'].shift(-7)  # Desplazar para predecir el cambio futuro


    return df

def prepare_datasets(df):
    """
    Dividir el dataset en train, val, test y preparar el DataLoader.
    """
    # Dividir en train, validation, test
    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)
    train_df, val_df = train_test_split(train_df, test_size=0.25, shuffle=False)  # 0.25 de 0.8 es el 20%

    # Definir un TimeSeriesDataSet para el modelo TFT
    max_encoder_length = 50  # Usar los últimos 60 días como entrada
    max_prediction_length = 7  # Predecir los próximos 7 días

    training = TimeSeriesDataSet(
        train_df,
        time_idx="timestamp",  # Índice temporal
        target="pct_cambio_7d",  # La variable de predicción
        group_ids=["timestamp"],  # Usamos timestamp como id de grupo
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=[],  # No usamos variables categóricas estáticas
        static_reals=[],  # No usamos variables estáticas continuas
        time_varying_known_categoricals=[],
        time_varying_known_reals=[col for col in df.columns if col != 'pct_cambio_7d'],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=["pct_cambio_7d"],  # La variable a predecir
        target_normalizer=GroupNormalizer(groups=["timestamp"], transformation="softplus"),  # Normalizar el target
    )

    validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=True, stop_randomization=True)
    test = TimeSeriesDataSet.from_dataset(training, test_df, predict=True, stop_randomization=True)

    return training, validation, test

def train_model(training, validation):
    """
    Entrenar el modelo TFT.
    """
    # Configurar el modelo Temporal Fusion Transformer
    trainer = Trainer(
        gpus=1,  # Usar GPU si está disponible
        max_epochs=30,  # Definir el número de épocas
    )

    tft = TemporalFusionTransformer.from_dataset(
        training,
        hidden_size=16,  # Tamaño de la capa oculta
        attention_head_size=4,  # Tamaño de las cabezas de atención
        dropout=0.1,  # Tasa de dropout
        hidden_continuous_size=8,  # Tamaño de las capas continuas
        output_size=7,  # Predecir los próximos 7 días
        loss=RMSE(),
    )

    # Entrenar el modelo
    trainer.fit(
        tft,
        train_dataloader=training,
        val_dataloaders=validation,
    )

    return tft

def evaluate_model(tft, test):
    """
    Evaluar el modelo en el conjunto de prueba.
    """
    raw_predictions, x = tft.predict(test, mode="raw", return_x=True)

    # Calcular las métricas de evaluación
    rmse = RMSE()(raw_predictions, x["target"])
    print(f"RMSE en el conjunto de prueba: {rmse.item()}")

def main(file_path):
    """
    Función principal que carga los datos, entrena el modelo y lo evalúa.
    """
    # Cargar y preprocesar los datos
    df = load_and_preprocess_data(file_path)

    # Preparar los datasets
    training, validation, test = prepare_datasets(df)

    # Entrenar el modelo
    tft = train_model(training, validation)

    # Evaluar el modelo
    evaluate_model(tft, test)

# Llamar a la función principal
if __name__ == "__main__":
    main(r"../../../../data/parquet/data_7d.parquet")
