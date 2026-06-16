# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 18:39:39 2026

@author: jeffs
"""

import pickle
import pandas as pd
import matplotlib.pyplot as plt

models = ['lstm', 'gru', 'informer', 'bostick']
data = []
for m in models:
    try:
        with open(f'{m}_validation_metrics.pkl', 'rb') as f:
            d = pickle.load(f)
            d['Model'] = m.upper()
            data.append(d)
        print(f"Cargado: {m}")
    except:
        print(f"No encontrado: {m}_validation_metrics.pkl")

df = pd.DataFrame(data).set_index('Model')
print(df)
df.to_csv('model_comparison.csv')

metrics = ['RMSE', 'R^2', 'SSIM']
plt.figure(figsize=(15, 5))
for i, met in enumerate(metrics):
    if met in df.columns:
        plt.subplot(1, 3, i+1)
        colors = ['green' if x == df[met].min() else 'gray' for x in df[met]] if met == 'RMSE' \
                 else ['green' if x == df[met].max() else 'gray' for x in df[met]]
        plt.bar(df.index, df[met], color=colors)
        plt.title(met)
        plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('model_comparison.png')
print("Comparación guardada en model_comparison.png y model_comparison.csv")