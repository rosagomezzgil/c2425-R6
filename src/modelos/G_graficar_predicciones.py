import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

def plot_predictions(y_real, pred = None, modelo = None, X = None):
    """
    Grafica predicciones junto con los valores reales.

    :return: Objeto de figura de matplotlib (fig)
    """

    try:
        timestamps = pd.to_datetime(y_real.index)
    except:
        print(type(y_real))
        print(y_real)
        raise Exception("")

    y_pred = pred if pred is not None else modelo.predict(X)

    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(timestamps, y_real, label="Real", color="blue")
    ax.plot(timestamps, y_pred, label="Predicción", color="red", linestyle="dashed")

    # Línea horizontal en y = 0
    ax.axhline(y=0, color="purple", linestyle="--", linewidth=1, label="y = 0")

    ax.legend()
    ax.set_xlabel("Tiempo")
    ax.set_ylabel(f"Valor")
    ax.set_title("Predicción vs Real")

    # Mostrar fecha cada 5 días
    locator = mdates.DayLocator(interval=30)
    formatter = mdates.DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return fig