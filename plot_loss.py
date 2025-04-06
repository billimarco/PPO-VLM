import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


# === SETUP ===
sns.set_style("white")
sns.set_context("talk")  # aumenta la leggibilità di etichette

# Lista dei file CSV
csv_files = ["entropy.csv", "exp_var.csv"]

# Colonne da tracciare specifiche per ogni file CSV
models_per_file = {
    "entropy.csv": {
        "CNN_Single_Frame": "Group: CNN_Single_Frame_Adapted - losses/entropy",
        "CNN_Multi_Frame": "Group: CNN_Adapted - losses/entropy",
        "ResNet18_Multi_Frame": "Group: ResNet_Adapted - losses/entropy",
        "ResNet18_Single_Frame": "Group: ResNet_Single_Frame_Adapted - losses/entropy",
        "Swin_Single_Frame": "Group: Swin_Single_Frame_Adapted_lrl-- - losses/entropy",
        "Swin_Single_Frame_LoRA": "Group: Swin_Single_Frame_Adapted_LoRA - losses/entropy",
    },
    "exp_var.csv": {
        "CNN_Single_Frame": "Group: CNN_Single_Frame_Adapted - losses/explained_variance",
        "CNN_Multi_Frame": "Group: CNN_Adapted - losses/explained_variance",
        "ResNet18_Multi_Frame": "Group: ResNet_Adapted - losses/explained_variance",
        "ResNet18_Single_Frame": "Group: ResNet_Single_Frame_Adapted - losses/explained_variance",
        "Swin_Single_Frame": "Group: Swin_Single_Frame_Adapted_lrl-- - losses/explained_variance",
        "Swin_Single_Frame_LoRA": "Group: Swin_Single_Frame_Adapted_LoRA - losses/explained_variance",
    },
}

# Dizionario per configurare titoli, etichette e colonne per ogni file CSV
file_config = {
    "entropy.csv": {
        "title": "Entropy",
        "ylabel": "Entropy",
    },
    "exp_var.csv": {
        "title": "Explained Variance",
        "ylabel": "Explained Variance",
    },
}

# Creazione della figura con 3 sottogruppi (subplots)
fig, axes = plt.subplots(2, 1, figsize=(7, 8))  # 2 riga, 1 colonne

# Loop sui file CSV
for idx, csv_file in enumerate(csv_files):
    # Carica il CSV
    df = pd.read_csv(csv_file)

    # Ottieni i modelli specifici per il file corrente
    models = models_per_file.get(csv_file, {})

    # Pulizia dati
    for model, col in models.items():
        for suffix in ["", "__MIN", "__MAX"]:
            col_with_suffix = col + suffix
            if col_with_suffix in df.columns:
                df[col_with_suffix] = pd.to_numeric(df[col_with_suffix], errors="coerce")

    # Colori distinti per ogni modello
    palette = sns.color_palette("tab10", n_colors=len(models))

    # Ottieni l'asse corrispondente
    ax = axes[idx]
    
    if csv_file == "exp_var.csv":
        # Creazione di un insetto (zoom) all'interno del grafico
        axins = inset_axes(ax, width=1.5, height=1.5, loc='lower right', bbox_to_anchor=(0.95, 0.05), bbox_transform=ax.transAxes)
        axins.set_xlim(1900000, df["global_step"].max())
        axins.set_ylim(0.8, 0.97)
        axins.grid(True, alpha=0.3)
        axins.set_xticks([])
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    # Traccia i dati
    for model, col in models.items():
        if col in df.columns:
            min_col = col + "__MIN"
            max_col = col + "__MAX"
            color = palette[list(models.keys()).index(model)]

            # Linea media
            ax.plot(
                df["global_step"],
                df[col],
                label=model,
                color=color,
                linewidth=2.5,
                alpha=1.0
            )
            
            if min_col in df.columns and max_col in df.columns:
                ax.fill_between(
                    df["global_step"],
                    df[min_col],
                    df[max_col],
                    color=color,
                    alpha=0.1
                )
                
            if csv_file == "exp_var.csv":
                axins.plot(df["global_step"], df[col], label=model, color=color, linewidth=2.5, alpha=1.0)

    # Configurazione specifica per il file corrente
    config = file_config.get(csv_file, {})
    ax.set_ylabel(config.get("ylabel", "Value"), fontsize=14)
    ax.grid(True, alpha=0.3)
    if csv_file == "entropy.csv":
        ax.legend(loc="upper right", fontsize=9, frameon=True)
    if csv_file == "exp_var.csv":
        ax.set_ylim(0, 1.0)
    ax.set_xlim(df["global_step"].min(), df["global_step"].max())
    
# Ottimizzazione layout
plt.tight_layout()

# Salva il grafico finale
plt.savefig("losses_plots.png")
plt.show()