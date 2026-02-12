import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import kendalltau
import seaborn as sns

# Data from the image
correlation_matrix = np.array([
    [1.,0.32, 0.58, 0.63, 0.52, 0.01],
    [0.32,1., 0.44, 0.46, 0.41, -0.15],
    [0.58,0.44, 1., 0.89, 0.79, -0.04],
    [0.63,0.46, 0.89, 1., 0.83, -0.03],
    [0.52,0.41, 0.79, 0.83, 1., -0.05],
    [0.01,-0.15, -0.04, -0.03, -0.05, 1.]
])

# Significance matrix (p-values)
pvalues_matrix = np.array([
    [0.,0.04, 0.04, 0.04, 0.04, 0.2],
    [0.04,0., 0.04, 0.04, 0.04, 0.2],
    [0.04,0.04, 0., 0.04, 0.04, 0.2],
    [0.04,0.04, 0.04, 0., 0.04, 0.2],
    [0.04,0.04, 0.04, 0.04, 0., 0.2],
    [0.2,0.2, 0.2, 0.2, 0.2, 0.]
])

# Mask for upper triangle
mask = np.tril(np.ones_like(correlation_matrix, dtype=bool))  # k=0 include la diagonale
variables_1 = ["Bruno", "Nicoletti", r'$rank$ by', r'$rank$ by', r'$rank$ by', r'$rank$ by']
variables_2 = ["et al.", "et al.", r'$\mathcal{P}$', r'$\mathcal{P}_{avg}$', r'$\mathcal{A}$', r'$\mathcal{C}$']

# Creazione della heatmap
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5, square=True,
            mask=mask, cbar_kws={"shrink": 0.5}, xticklabels=False, yticklabels=True, annot_kws={"color": "black"})

#plt.xticks(np.arange(len(variables)), variables, rotation=45)
#plt.yticks(np.arange(len(variables_names)), variables_names) 

# plt.title("Pearson Correlation Matrix", fontsize=11, pad=10)  

# Spostare le etichette della x in alto
#ax.xaxis.tick_top()
#ax.xaxis.set_label_position('top')
#plt.xticks(rotation=45, ha='left')

# non mostrare le etichette dell'asse X
ax.set_xticklabels([])

# Rimozione delle tacche dell'asse Y
ax.set_yticks([])
ax.tick_params(axis='y', bottom=False, top=False)
ax.tick_params(axis='x', bottom=False, top=False)



# Aggiungere i p-value sopra i quadrati della heatmap
for i in range(len(variables_1)):
    for j in range(len(variables_1)):
        if i < j:  # Solo triangolo superiore
            p_val = pvalues_matrix[i][j]
            if p_val < 0.05:  # Mostra solo p-value significativi
                text_color = "black"
                ax.text(j + 0.5, i + 0.17, f"p < 0.05", ha='center', va='center', fontsize=8, color=text_color)
            else:
                text_color = "black"
                ax.text(j + 0.5, i + 0.17, f"p ≥ 0.05", ha='center', va='center', fontsize=8, color=text_color)

# Scrivi le etichette delle variabili sulla diagonale (al posto dei valori)
for i, label in enumerate(variables_2):
    ax.text(i + 0.5, i + 0.3, variables_1[i], ha='center', va='center', fontsize=10, color="black")
    ax.text(i + 0.5, i + 0.6, label, ha='center', va='center', fontsize=10, color="black")

# Rimuovi le yticklabels e xticklabels per evitare doppioni
ax.set_yticklabels([])
ax.set_xticklabels([])

# Spazio regolato
plt.subplots_adjust(left=0.2, right=0.9, top=0.823, bottom=0.2)

# Salvataggio e visualizzazione
plt.savefig(f"_tentative_scripts/img/heatmap_kendalltau.png", dpi=300, bbox_inches='tight')
plt.show()

