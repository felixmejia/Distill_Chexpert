import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, hamming_loss, jaccard_score
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import TARGET_CLASSES_5

import os

def calculate_metrics(y_true, y_pred_probs, class_names):
    """Calcula un conjunto de métricas para clasificación multietiqueta."""
    metrics = {}
    y_pred_binary = (y_pred_probs > 0.5).int()
    
    # Métricas por clase
    for i, name in enumerate(class_names):
        try:
            metrics[f'auc_{name}'] = roc_auc_score(y_true[:, i], y_pred_probs[:, i])
        except ValueError:
            metrics[f'auc_{name}'] = 0.5 # Si solo hay una clase presente
        
        # Calcular F1, Precisión y Recall, con 'zero_division=0' para evitar warnings
        # si no hay predicciones o etiquetas positivas.
        metrics[f'f1_{name}'] = f1_score(y_true[:, i], y_pred_binary[:, i], zero_division=0)
        metrics[f'precision_{name}'] = precision_score(y_true[:, i], y_pred_binary[:, i], zero_division=0)
        metrics[f'recall_{name}'] = recall_score(y_true[:, i], y_pred_binary[:, i], zero_division=0)
        metrics[f'accuracy_{name}'] = accuracy_score(y_true[:, i], y_pred_binary[:, i])


    # Métricas promedio (Macro)
    metrics['auc_macro_avg'] = np.mean([metrics[f'auc_{name}'] for name in class_names])
    metrics['f1_macro_avg'] = np.mean([metrics[f'f1_{name}'] for name in class_names])
    metrics['precision_macro_avg'] = np.mean([metrics[f'precision_{name}'] for name in class_names])
    metrics['recall_macro_avg'] = np.mean([metrics[f'recall_{name}'] for name in class_names])
    metrics['accuracy_macro_avg'] = np.mean([metrics[f'accuracy_{name}'] for name in class_names])
    
    # --- Métricas de Conjunto (Multi-Label) ---
    metrics['hamming_loss'] = hamming_loss(y_true, y_pred_binary)
    metrics['jaccard_samples_avg'] = jaccard_score(y_true, y_pred_binary, average='samples', zero_division=1)
   

    return metrics


def plot_metric_learning_curves(history, class_names, metric_name, fold, model_name, results_dir):
    """
    Grafica la evolución de una métrica específica (AUC, F1, etc.) para cada patología.
    """
    metric_key_map = {
        "AUC": "auc",
        "F1-Score": "f1",
        "Precision": "precision",
        "Recall": "recall",
        "Accuracy": "accuracy"  
    }
    metric_prefix = metric_key_map.get(metric_name)
    if not metric_prefix:
        print(f"ADVERTENCIA: Métrica '{metric_name}' no reconocida para graficar.")
        return

    plt.figure(figsize=(12, 7))
    
    for name in class_names:
        # Construye la clave para buscar en el diccionario, ej. 'f1_Cardiomegaly'
        full_metric_key = f'{metric_prefix}_{name}'
        # Extrae los scores de esta métrica a lo largo de todas las épocas
        metric_scores = [epoch_metrics.get(full_metric_key, 0) for epoch_metrics in history['val_metrics']]
        plt.plot(metric_scores, label=f'{metric_name} {name}')
    
    plt.title(f'Curvas de Aprendizaje por Patología ({metric_name}) - {model_name} - Fold {fold+1}')
    plt.xlabel('Época'); plt.ylabel(f'{metric_name} Score'); plt.legend(loc='lower right'); plt.grid(True)
    
    # Ajustar el eje Y para una mejor visualización según la métrica
    if metric_name == "AUC":
        plt.ylim(0.4, 1.0)
    else:
        plt.ylim(0.0, 1.0)
    
    # Guardar con un nombre de archivo descriptivo
    filename = f'{model_name}_fold_{fold+1}_pathology_{metric_prefix}_curves.png'
    plt.savefig(os.path.join(results_dir, filename))
    plt.close()


def plot_learning_curves(history, fold, model_name, results_dir):
    """Grafica las curvas de pérdida de entrenamiento y validación."""
    plt.figure(figsize=(10, 5))
    
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    
    plt.title(f'Curvas de Pérdida (Loss) - {model_name} - Fold {fold+1}')
    plt.xlabel('Época'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(results_dir, f'{model_name}_fold_{fold+1}_loss_curve.png'))
    plt.close()



def plot_per_pathology_learning_curves(history, class_names, fold, model_name, results_dir):
    """Grafica la evolución del AUC para cada patología."""
    plt.figure(figsize=(12, 7))
    for name in class_names:
        auc_scores = [epoch_metrics[f'auc_{name}'] for epoch_metrics in history['val_metrics']]
        plt.plot(auc_scores, label=f'AUC {name}')
    
    plt.title(f'Curvas de Aprendizaje por Patología (AUC) - {model_name} - Fold {fold+1}')
    plt.xlabel('Época'); plt.ylabel('AUC Score'); plt.legend(loc='lower right'); plt.grid(True)
    plt.ylim(0.5, 1.0)
    plt.savefig(os.path.join(results_dir, f'{model_name}_fold_{fold+1}_pathology_auc_curves.png'))
    plt.close()

def plot_roc_curves(y_true, y_pred_probs, class_names, model_name, results_dir):
    """Grafica las curvas ROC para cada patología y una macro-promediada."""
    plt.figure(figsize=(12, 10))
    
    # Calcular y graficar para cada clase
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC {name} (AUC = {roc_auc:.3f})')

    # Calcular y graficar ROC Macro-Average
    # Primero, se agregan todos los FPRs
    all_fpr = np.unique(np.concatenate([roc_curve(y_true[:, i], y_pred_probs[:, i])[0] for i in range(len(class_names))]))
    # Luego, se interpolan las curvas ROC en estos puntos
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_probs[:, i])
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    # Finalmente, se promedia y se calcula el AUC
    mean_tpr /= len(class_names)
    macro_roc_auc = auc(all_fpr, mean_tpr)
    plt.plot(all_fpr, mean_tpr,
             label=f'Macro-Avg ROC (AUC = {macro_roc_auc:.3f})',
             color='navy', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', label='Azar')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR)'); plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title(f'Curvas AUC-ROC - Modelo {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f'{model_name}_final_roc_curves.png'))
    plt.close()
    print(f"Gráfica de curvas ROC guardada en {results_dir}")


def plot_distribution_stacked_bar(train_df, val_df, class_names, results_dir):
    """
    Genera una gráfica de barras apiladas mostrando la distribución de patologías
    en los conjuntos de entrenamiento y validación para un fold específico.
    """
    # Contar los casos positivos para cada patología (donde el valor es 1)
    train_counts = train_df[class_names].apply(lambda x: (x == 1).sum())
    val_counts = val_df[class_names].apply(lambda x: (x == 1).sum())

    # Crear un DataFrame para la gráfica
    plot_df = pd.DataFrame({
        'Entrenamiento': train_counts,
        'Validación (del fold)': val_counts
    })

    # Graficar
    plot_df.sort_values(by='Entrenamiento', ascending=False).plot(
        kind='bar',
        stacked=True,
        figsize=(14, 8),
        colormap='viridis'  # Paleta de colores para buena visualización
    )

    plt.title(f'Distribución de Clases para DataSet ChexPert')
    plt.ylabel('Número de Casos Positivos')
    plt.xlabel('Patologías')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()  # Ajustar para que las etiquetas no se corten
    
    filename = f'ChexPert_dataset_distribution.png'
    plt.savefig(os.path.join(results_dir, filename))
    plt.close()
    print(f"INFO: Gráfica de distribución del dataset para el train guardada en '{filename}'.")


def plot_fold_distribution_stacked_bar(train_df, val_df, class_names, fold_idx, results_dir):
    """
    Genera una gráfica de barras apiladas mostrando la distribución de patologías
    en los conjuntos de entrenamiento y validación para un fold específico.
    """
    # Contar los casos positivos para cada patología (donde el valor es 1)
    train_counts = train_df[class_names].apply(lambda x: (x == 1).sum())
    val_counts = val_df[class_names].apply(lambda x: (x == 1).sum())

    # Crear un DataFrame para la gráfica
    plot_df = pd.DataFrame({
        'Entrenamiento': train_counts,
        'Validación (del fold)': val_counts
    })

    # Graficar
    plot_df.sort_values(by='Entrenamiento', ascending=False).plot(
        kind='bar',
        stacked=True,
        figsize=(14, 8),
        colormap='viridis'  # Paleta de colores para buena visualización
    )

    plt.title(f'Distribución de Clases para el Fold {fold_idx + 1}')
    plt.ylabel('Número de Casos Positivos')
    plt.xlabel('Patologías')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()  # Ajustar para que las etiquetas no se corten
    
    filename = f'fold_{fold_idx+1}_dataset_distribution_valid.png'
    plt.savefig(os.path.join(results_dir, filename))
    plt.close()
    print(f"INFO: Gráfica de distribución del dataset para el Fold {fold_idx+1} guardada en '{filename}'.")


def plot_dataset_distribution(df, target_cols, file_path):
    """Genera un gráfico de barras apiladas de la distribución de clases."""
    # Esta es una simplificación, se debe adaptar al formato train/test real
    plt.figure(figsize=(15, 8))
    df[target_cols].sum().sort_values(ascending=False).plot(kind='bar')
    plt.title('Distribución de Clases en el Dataset de Entrenamiento')
    plt.ylabel('Número de Casos Positivos')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()


def create_summary_table(all_results):
    """Crea una tabla resumen con medias y desviaciones estándar."""
    summary_data = []
    for model_name, folds_data in all_results.items():
        # Extraer métricas de cada fold
        auc_scores = [fold['metrics']['auc_macro'] for fold in folds_data]
        f1_scores = [fold['metrics']['f1_macro'] for fold in folds_data]
        exec_times = [fold['time'] for fold in folds_data]

        summary_data.append({
            'Model': model_name,
            'Mean AUC': np.mean(auc_scores),
            'Std AUC': np.std(auc_scores),
            'Mean F1-Score': np.mean(f1_scores),
            'Std F1-Score': np.std(f1_scores),
            'Mean Exec Time (s)': np.mean(exec_times),
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\n--- Tabla Resumen de Resultados ---")
    print(summary_df.to_string(index=False))
    summary_df.to_csv('results/summary_results.csv', index=False)




def plot_stacked_bars_original_states(df, title_suffix, pathologies, output_dir):
    """
    Generates a horizontal stacked bar chart for the 4 original states
    (Present, Absent, Uncertain, Not Mentioned) for each pathology.
    """
    state_counts_per_pathology = {}
    state_order = ['Present', 'Absent', 'Uncertain', 'Not Mentioned']
    # Mapeo de valores numéricos a nombres de estado para consistencia
    value_to_state_map = {
        1.0: 'Present',
        0.0: 'Absent',
        -1.0: 'Uncertain'
        # NaN se manejará por separado
    }

    for p in pathologies:
        # Contar cada estado, incluyendo NaN como 'Not Mentioned'
        counts = df[p].value_counts(dropna=False).rename(value_to_state_map)
        # Asegurar que todos los estados estén presentes, incluso con conteo 0
        current_pathology_counts = {state: 0 for state in state_order}
        for state_name, count_val in counts.items():
            if pd.isna(state_name): # Si el índice es NaN (originalmente NaN en value_counts)
                current_pathology_counts['Not Mentioned'] += count_val
            elif state_name in current_pathology_counts: # Si es uno de los estados mapeados
                current_pathology_counts[state_name] += count_val
        state_counts_per_pathology[p] = current_pathology_counts

    # Convertir a DataFrame para facilitar el ploteo
    plot_df = pd.DataFrame(state_counts_per_pathology).T # Transponer para tener patologías como filas
    plot_df = plot_df[state_order] # Asegurar el orden de las columnas/segmentos

    ax = plot_df.plot(kind='barh', stacked=True, figsize=(12, 8),
                      color={'Present': 'forestgreen', 'Absent': 'lightcoral',
                             'Uncertain': 'gold', 'Not Mentioned': 'lightgray'})

    # Añadir etiquetas a cada segmento
    for p_idx, pathology_name in enumerate(plot_df.index):
        cumulative_width = 0
        for state_idx, state_name in enumerate(plot_df.columns):
            count = plot_df.loc[pathology_name, state_name]
            if count > 0: # Solo anotar si el conteo es mayor a 0
                # El centro del segmento actual
                center_x = cumulative_width + count / 2
                ax.text(center_x, p_idx, str(int(count)),
                        ha='center', va='center', color='black', fontsize=9)
            cumulative_width += count

    plt.title(f'Distribution of Pathological States ({title_suffix}) - Original')
    plt.xlabel('Number of records')
    plt.ylabel('Pathologies')
    plt.legend(title='State', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Ajustar para que la leyenda quepa
    output_path = os.path.join(output_dir, f'{title_suffix.lower().replace(" ", "_")}_4_states_stacked_bar.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Four state chart for '{title_suffix}' save in: {output_path}")


def plot_stacked_bars_combined_states(df, title_suffix, pathologies, output_dir):
    """
    Generates a horizontal stacked bar chart for 2 combined states:
    - Positive Combined: Present (1.0) OR Uncertain (-1.0)
    - Negative Combined: Absent (0.0) OR Not Mentioned (NaN)
    """
    combined_state_counts = {}
    state_order = ['Present-Uncertain', 'Absent-Not Mentioned']

    for p in pathologies:
        positive_count = df[p].isin([1.0, -1.0]).sum()
        # Negative_count se puede calcular como total - positive, o sumando 0.0 y NaN
        negative_count = df[p].isin([0.0]).sum() + df[p].isna().sum()

        combined_state_counts[p] = {
            'Present-Uncertain': positive_count,
            'Absent-Not Mentioned': negative_count
        }

    plot_df = pd.DataFrame(combined_state_counts).T
    plot_df = plot_df[state_order] # Asegurar el orden

    ax = plot_df.plot(kind='barh', stacked=True, figsize=(12, 8),
                      color={'Present-Uncertain': 'mediumseagreen', 'Absent-Not Mentioned': 'salmon'})

    # Añadir etiquetas
    for p_idx, pathology_name in enumerate(plot_df.index):
        cumulative_width = 0
        for state_idx, state_name in enumerate(plot_df.columns):
            count = plot_df.loc[pathology_name, state_name]
            if count > 0:
                center_x = cumulative_width + count / 2
                ax.text(center_x, p_idx, str(int(count)),
                        ha='center', va='center', color='black', fontsize=9)
            cumulative_width += count

    plt.title(f'Distribution of Pathological States  ({title_suffix}) - Grouped States')
    plt.xlabel('Number of records')
    plt.ylabel('Pathologies')
    plt.legend(title='Grouped States', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    output_path = os.path.join(output_dir, f'{title_suffix.lower().replace(" ", "_")}_2_states_stacked_bar.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Two state chart for  '{title_suffix}' save in: {output_path}")



