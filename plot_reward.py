import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import numpy as np
from matplotlib.ticker import MultipleLocator

# === SETUP ===
sns.set_style("white")
sns.set_context("talk")  # aumenta la leggibilità di etichette

# Lista dei file CSV
csv_files = ["reward.csv", "episode_len.csv", "kills.csv"]

# Colonne da tracciare specifiche per ogni file CSV
models_per_file = {
    "reward.csv": {
        "CNN_Single_Frame": "Group: CNN_Single_Frame_Adapted - Default-Conf-v1/reward",
        "CNN_Multi_Frame": "Group: CNN_Adapted - Default-Conf-v1/reward",
        "ResNet18_Multi_Frame": "Group: ResNet_Adapted - Default-Conf-v1/reward",
        "ResNet18_Single_Frame": "Group: ResNet_Single_Frame_Adapted - Default-Conf-v1/reward",
        "Swin_Single_Frame": "Group: Swin_Single_Frame_Adapted_lrl-- - Default-Conf-v1/reward",
        "Swin_Single_Frame_LoRA": "Group: Swin_Single_Frame_Adapted_LoRA - Default-Conf-v1/reward",
    },
    "episode_len.csv": {
        "CNN_Single_Frame": "Group: CNN_Single_Frame_Adapted - Default-Conf-v1/episode_len",
        "CNN_Multi_Frame": "Group: CNN_Adapted - Default-Conf-v1/episode_len",
        "ResNet18_Multi_Frame": "Group: ResNet_Adapted - Default-Conf-v1/episode_len",
        "ResNet18_Single_Frame": "Group: ResNet_Single_Frame_Adapted - Default-Conf-v1/episode_len",
        "Swin_Single_Frame": "Group: Swin_Single_Frame_Adapted_lrl-- - Default-Conf-v1/episode_len",
        "Swin_Single_Frame_LoRA": "Group: Swin_Single_Frame_Adapted_LoRA - Default-Conf-v1/episode_len",
    },
    "kills.csv": {
        "CNN_Single_Frame": "Group: CNN_Single_Frame_Adapted - Default-Conf-v1/kills",
        "CNN_Multi_Frame": "Group: CNN_Adapted - Default-Conf-v1/kills",
        "ResNet18_Multi_Frame": "Group: ResNet_Adapted - Default-Conf-v1/kills",
        "ResNet18_Single_Frame": "Group: ResNet_Single_Frame_Adapted - Default-Conf-v1/kills",
        "Swin_Single_Frame": "Group: Swin_Single_Frame_Adapted_lrl-- - Default-Conf-v1/kills",
        "Swin_Single_Frame_LoRA": "Group: Swin_Single_Frame_Adapted_LoRA - Default-Conf-v1/kills",
    },
}

# Dizionario per configurare titoli, etichette e colonne per ogni file CSV
file_config = {
    "reward.csv": {
        "title": "Model Reward",
        "ylabel": "Reward",
    },
    "episode_len.csv": {
        "title": "Episode Length",
        "ylabel": "Episode Length",
    },
    "kills.csv": {
        "title": "Kills",
        "ylabel": "Kills",
    },
}

# Creazione della figura con 3 sottogruppi (subplots)
fig, axes = plt.subplots(3, 1, figsize=(7, 12))  # 3 riga, 1 colonne

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

    # Configurazione specifica per il file corrente
    config = file_config.get(csv_file, {})
    ax.set_ylabel(config.get("ylabel", "Value"), fontsize=14)
    ax.xaxis.set_major_locator(MultipleLocator(200000))
    ax.grid(True, alpha=0.3)
    if csv_file == "reward.csv":
        ax.legend(loc="upper left", fontsize=9, frameon=True)
    ax.set_xlim(df["global_step"].min(), df["global_step"].max())
    
# Ottimizzazione layout
plt.tight_layout()

# Salva il grafico finale
plt.savefig("reward_plots.png")
plt.show()