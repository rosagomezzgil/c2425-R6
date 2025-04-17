import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df= pd.read_csv(f"../../../data/limpieza/cleaned_data.csv")
df["pct_cambio_30d"].hist(bins=50)
plt.show()