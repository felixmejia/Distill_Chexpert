import torch
import time
import os
import pandas as pd
from codecarbon import EmissionsTracker

from dataset import create_train_dataloaders, create_final_train_dataloader, get_holdout_validation_dataloader, TARGET_CLASSES_5
from models import get_teacher_model, get_student_model
from trainer import train_one_epoch, validate_one_epoch, save_checkpoint, load_checkpoint
from distillation import DistillationLoss
from utils import plot_learning_curves, plot_per_pathology_learning_curves, plot_roc_curves, calculate_metrics

from tqdm import tqdm

# --- Configuración Global del Experimento ---
CONFIG = {
    "data_path": "/workspace/WORKS/DATA/CheXpert-v1.0-small",
    "k_folds": 3,  # Usar 1 fold para demostraciones rápidas, 3 o 5 para robustez
    "num_epochs": 50, # Número de épocas para cada entrenamiento
    "batch_size": 64,
    "learning_rate": 1e-4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "results_dir": "results_12062025_1",
    # --- NUEVO: Fracción de datos para pruebas rápidas (1.0 para usar todos los datos) ---
    "data_fraction": 1.0, 
    # --- Parámetros de Destilación ---
    "alpha": 0.3, # Peso para la 'hard loss' (etiquetas reales)
    "temperature": 5.0, # Temperatura para suavizar los logits
}



def print_config(config_dict):
    """Imprime la configuración del experimento en un formato de tabla legible."""
    print("\n" + "="*50)
    print("⚙️  PARÁMETROS DE LA EJECUCIÓN  ⚙️")
    print("="*50)
    for key, value in config_dict.items():
        print(f"{key:<20}: {value}")
    print("="*50 + "\n")


def report_cumulative_emissions(csv_path):
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)
    if not df.empty:
        total_energy_kwh = df['energy_consumed'].sum()
        total_emissions_kg = df['emissions'].sum()
        print("\n--- 🔋 Impacto Ambiental Acumulado (Todas las Ejecuciones) ---")
        print(f"Consumo Energético Total: {total_energy_kwh:.6f} kWh")
        print(f"Emisiones de CO2 Totales: {total_emissions_kg:.6f} kg")

def run_experiment(model_name, model, train_loader, val_loader, fold_idx, is_distillation=False, teacher_model=None):
    print(f"\n--- Entrenando: {model_name} | Fold: {fold_idx+1}/{CONFIG['k_folds']} ---")
    criterion = torch.nn.BCEWithLogitsLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    checkpoint_path = os.path.join(CONFIG['results_dir'], f"{model_name}_fold_{fold_idx}_checkpoint.pth")
    start_epoch, best_val_auc, history = load_checkpoint(checkpoint_path, model, optimizer)
    
    model.to(CONFIG['device'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor): state[k] = v.to(CONFIG['device'])

    loss_fn_kd = None
    if is_distillation and teacher_model:
        teacher_model.to(CONFIG['device']).eval()
        loss_fn_kd = DistillationLoss(criterion, teacher_model, CONFIG['alpha'], CONFIG['temperature'])

    for epoch in range(start_epoch, CONFIG['num_epochs']):
        train_desc = f"Train {model_name} [Fold {fold_idx+1}, Epoch {epoch+1}/{CONFIG['num_epochs']}]"
        val_desc = f"Valid {model_name} [Fold {fold_idx+1}, Epoch {epoch+1}/{CONFIG['num_epochs']}]"
        
        train_one_epoch(model, train_loader, criterion, optimizer, CONFIG['device'], train_desc, loss_fn_kd)
        val_loss, val_metrics = validate_one_epoch(model, val_loader, criterion, CONFIG['device'], TARGET_CLASSES_5, val_desc)
        
        history['train_loss'].append(val_loss)
        history['val_loss'].append(val_loss)
        history['val_metrics'].append(val_metrics)
        
        current_auc = val_metrics['auc_macro_avg']
        print(f"Epoch {epoch+1} | Val Loss: {val_loss:.4f} | Val AUC (Macro Avg): {current_auc:.4f}")

        if current_auc > best_val_auc:
            best_val_auc = current_auc
            save_checkpoint(epoch, model, optimizer, best_val_auc, history, checkpoint_path)
            print(f"✅ Nuevo mejor checkpoint guardado para {model_name} (Fold {fold_idx+1}) con AUC: {best_val_auc:.4f}")

    plot_learning_curves(history, fold_idx, model_name, CONFIG['results_dir'])
    plot_per_pathology_learning_curves(history, TARGET_CLASSES_5, fold_idx, model_name, CONFIG['results_dir'])
    
    # Devolver las métricas del mejor punto para este fold
    best_metrics = max(history['val_metrics'], key=lambda x: x['auc_macro_avg'])
    return best_metrics


def run_experiment_old(model_name, model, dataloaders, is_distillation=False, teacher_model=None):
    print(f"\n{'='*20} Iniciando Experimento: {model_name} {'='*20}")
    
    criterion = torch.nn.BCEWithLogitsLoss()
    fold_idx = 0  # Simplificado a un solo fold
    train_loader, val_loader = dataloaders[fold_idx]
    
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    checkpoint_path = os.path.join(CONFIG['results_dir'], f"{model_name}_checkpoint.pth")
    start_epoch, best_val_auc, history = load_checkpoint(checkpoint_path, model, optimizer)
    
    model.to(CONFIG['device'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor): state[k] = v.to(CONFIG['device'])

    # --- Configurar pérdida de destilación si es necesario ---
    loss_fn_kd = None
    if is_distillation and teacher_model:
        print(f"INFO: Configurando entrenamiento con destilación. Alpha={CONFIG['alpha']}, Temp={CONFIG['temperature']}")
        teacher_model.to(CONFIG['device']).eval()
        loss_fn_kd = DistillationLoss(criterion, teacher_model, CONFIG['alpha'], CONFIG['temperature'])

    for epoch in range(start_epoch, CONFIG['num_epochs']):
        # --- NUEVO: Descripciones dinámicas para la barra de progreso ---
        train_desc = f"Train {model_name} [Epoch {epoch+1}/{CONFIG['num_epochs']}]"
        val_desc = f"Valid {model_name} [Epoch {epoch+1}/{CONFIG['num_epochs']}]"
        
        train_one_epoch(model, train_loader, criterion, optimizer, CONFIG['device'], train_desc, loss_fn_kd)
        val_loss, val_metrics = validate_one_epoch(model, val_loader, criterion, CONFIG['device'], TARGET_CLASSES_5, val_desc)
        
        history['train_loss'].append(val_loss) # Simplificación, normalmente se calcula pérdida de train
        history['val_loss'].append(val_loss)
        history['val_metrics'].append(val_metrics)
        
        current_auc = val_metrics['auc_macro_avg']
        print(f"Epoch {epoch+1} | Val Loss: {val_loss:.4f} | Val AUC (Macro Avg): {current_auc:.4f}")

        if current_auc > best_val_auc:
            best_val_auc = current_auc
            save_checkpoint(epoch, model, optimizer, best_val_auc, history, checkpoint_path)
            print(f"✅ Nuevo mejor checkpoint guardado para {model_name} con AUC: {best_val_auc:.4f}")

    print(f"\n📊 Generando gráficas de aprendizaje para {model_name}...")
    plot_learning_curves(history, fold_idx, model_name, CONFIG['results_dir'])
    plot_per_pathology_learning_curves(history, TARGET_CLASSES_5, fold_idx, model_name, CONFIG['results_dir'])
    
    print(f"🏁 Entrenamiento completado para {model_name}.")
    return checkpoint_path

def evaluate_model_on_holdout(model_name, model_class, checkpoint_path, holdout_loader):
    print(f"\n--- 🧪 Evaluando {model_name} en el Hold-Out Set (`valid.csv`) ---")
    model = model_class(num_classes=len(TARGET_CLASSES_5))
    checkpoint = torch.load(checkpoint_path, map_location=CONFIG['device'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(CONFIG['device']).eval()
    
    all_labels, all_preds = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(holdout_loader, desc=f"Evaluando {model_name}"):
            outputs = model(inputs.to(CONFIG['device']))
            all_labels.append(labels)
            all_preds.append(torch.sigmoid(outputs.cpu()))
    
    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)
    
    metrics = calculate_metrics(all_labels, all_preds, TARGET_CLASSES_5)
    plot_roc_curves(all_labels, all_preds, TARGET_CLASSES_5, model_name, CONFIG['results_dir'])
    
    # Imprimir la tabla de métricas detallada
    print(f"\n--- Métricas Detalladas para: {model_name} ---")
    header = f"{'Patología':<18} | {'AUC':<6} | {'F1':<6} | {'Precision':<9} | {'Recall':<6}"
    print(header)
    print('-' * len(header.replace('|', '+')))
    for pathology in TARGET_CLASSES_5:
        auc = metrics.get(f'auc_{pathology}', 0)
        f1 = metrics.get(f'f1_{pathology}', 0)
        precision = metrics.get(f'precision_{pathology}', 0)
        recall = metrics.get(f'recall_{pathology}', 0)
        print(f"{pathology:<18} | {auc:.4f} | {f1:.4f} | {precision:.4f}  | {recall:.4f}")
    
    # Devolver un diccionario con los promedios para la tabla resumen
    return {
        "Modelo": model_name,
        "AUC Macro": metrics['auc_macro_avg'],
        "Precision Macro": metrics['precision_macro_avg'],
        "Recall Macro": metrics['recall_macro_avg']
    }

if __name__ == '__main__':
    os.makedirs(CONFIG['results_dir'], exist_ok=True)
    
    tracker = EmissionsTracker(output_dir=CONFIG['results_dir'], project_name="Full_CheXpert_KD_Pipeline")
    tracker.start()
    
    print("Dispositivo de cómputo:", CONFIG['device'])
    

    print_config(CONFIG)

    # 1. Cargar Datos
    dataloaders_kfold = create_train_dataloaders(CONFIG['data_path'], CONFIG['k_folds'], CONFIG['batch_size'], CONFIG['data_fraction'])
    

    # 2. Bucle Principal de K-Fold
    all_fold_results = {
        "Teacher": [],
        "Student_Baseline": [],
        "Student_KD": []
    }

    for fold_idx, (train_loader, val_loader) in enumerate(dataloaders_kfold):
        print(f"\n{'='*25} INICIANDO FOLD {fold_idx + 1}/{CONFIG['k_folds']} {'='*25}")

        # --- Entrenar Teacher para este fold ---
        teacher_model = get_teacher_model(num_classes=len(TARGET_CLASSES_5))
        teacher_metrics = run_experiment("Teacher", teacher_model, train_loader, val_loader, fold_idx)
        all_fold_results["Teacher"].append(teacher_metrics)
        # Cargar el mejor estado del teacher para la destilación en este fold
        teacher_model.load_state_dict(torch.load(os.path.join(CONFIG['results_dir'], f"Teacher_fold_{fold_idx}_checkpoint.pth"))['model_state_dict'])

        # --- Entrenar Student Baseline para este fold ---
        student_base_model = get_student_model(num_classes=len(TARGET_CLASSES_5))
        student_base_metrics = run_experiment("Student_Baseline", student_base_model, train_loader, val_loader, fold_idx)
        all_fold_results["Student_Baseline"].append(student_base_metrics)

        # --- Entrenar Student con Destilación para este fold ---
        student_kd_model = get_student_model(num_classes=len(TARGET_CLASSES_5))
        student_kd_metrics = run_experiment("Student_KD", student_kd_model, train_loader, val_loader, fold_idx, is_distillation=True, teacher_model=teacher_model)
        all_fold_results["Student_KD"].append(student_kd_metrics)

    
    
    
    # 3. Procesar Resultados y Mostrar Tabla Resumen con Desviación Estándar
    print("\n\n" + "="*50 + "\n🎉 RESUMEN DE VALIDACIÓN CRUZADA (K-FOLD) 🎉\n" + "="*50)
    
    summary_data = []
    for model_name, fold_metrics_list in all_fold_results.items():
        aucs = [m['auc_macro_avg'] for m in fold_metrics_list]
        precisions = [m['precision_macro_avg'] for m in fold_metrics_list]
        recalls = [m['recall_macro_avg'] for m in fold_metrics_list]
        
        summary_data.append({
            "Modelo": model_name,
            "AUC Media": np.mean(aucs),
            "AUC Std Dev": np.std(aucs),
            "Precision Media": np.mean(precisions),
            "Recall Media": np.mean(recalls)
        })
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))


    # FASE 2
    print("\n\n" + "#"*60 + "\n# FASE 2: INICIANDO RE-ENTRENAMIENTO FINAL CON TODOS LOS DATOS\n" + "#"*60)
    final_train_loader = create_final_train_dataloader(CONFIG['data_path'], CONFIG['batch_size'], CONFIG['data_fraction'])
    final_teacher_model = get_teacher_model(num_classes=len(TARGET_CLASSES_5)); final_teacher_ckpt = train_final_model("Teacher", final_teacher_model, final_train_loader)
    final_teacher_model.load_state_dict(torch.load(final_teacher_ckpt))
    final_student_base_model = get_student_model(num_classes=len(TARGET_CLASSES_5)); final_student_base_ckpt = train_final_model("Student_Baseline", final_student_base_model, final_train_loader)
    final_student_kd_model = get_student_model(num_classes=len(TARGET_CLASSES_5)); final_student_kd_ckpt = train_final_model("Student_KD", final_student_kd_model, final_train_loader, is_distillation=True, teacher_model=final_teacher_model)
    
    # FASE 3
    print("\n\n" + "#"*60 + "\n# FASE 3: INICIANDO EVALUACIÓN DEFINITIVA SOBRE `valid.csv`\n" + "#"*60)
    holdout_loader = get_holdout_validation_dataloader(CONFIG['data_path'], CONFIG['batch_size'], CONFIG['data_fraction'])
    final_results = []
    models_to_evaluate = {
        "Teacher Final": (get_teacher_model, final_teacher_ckpt), "Student Baseline Final": (get_student_model, final_student_base_ckpt), "Student KD Final": (get_student_model, final_student_kd_ckpt)
    }
    for name, (model_fn, ckpt) in models_to_evaluate.items():
        avg_metrics = evaluate_model_on_holdout(name, model_fn, ckpt, holdout_loader); final_results.append(avg_metrics)
    
    print("\n\n" + "="*60 + "\n🎉 RESULTADOS FINALES Y DEFINITIVOS (PROMEDIOS Y MÉTRICAS GLOBALES) 🎉\n" + "="*60)
    display_cols = {"Modelo": "Modelo", "auc_macro_avg": "AUC Macro", "accuracy_macro_avg": "Accuracy Macro", "jaccard_samples_avg": "Jaccard Score (IoU)", "hamming_loss": "Hamming Loss (Error)"}
    final_df = pd.DataFrame(final_results)
    final_df_display = final_df[list(display_cols.keys())].rename(columns=display_cols)
    print(final_df_display.to_string(index=False))


    # 4. Evaluación Final en Hold-Out Set (usando el modelo del fold 0 como representante)
    print("\n\n" + "="*50 + "\n🧪 EVALUACIÓN FINAL SOBRE `valid.csv` (Usando Modelo de Fold 0) 🧪\n" + "="*50)
    holdout_loader = get_holdout_validation_dataloader(CONFIG['data_path'], CONFIG['batch_size'], CONFIG['data_fraction'])
    
    # Cargar y evaluar el Teacher del fold 0
    teacher_model_final = get_teacher_model(num_classes=len(TARGET_CLASSES_5))
    teacher_ckpt_path = os.path.join(CONFIG['results_dir'], "Teacher_fold_0_checkpoint.pth")
    # (El código de evaluación detallado por patología ya está en la función, aquí solo la llamamos)
    evaluate_model_on_holdout("Teacher", get_teacher_model, teacher_ckpt_path, holdout_loader) # Descomentar si se desea la evaluación detallada final

    tracker.stop()
    report_cumulative_emissions(os.path.join(CONFIG['results_dir'], 'emissions.csv'))


    # Se definen los 3 modelos finales a evaluar
    print("\n\n" + "#"*60 + "\n# FASE 3: INICIANDO EVALUACIÓN DEFINITIVA SOBRE `valid.csv`\n" + "#"*60)
    holdout_loader = get_holdout_validation_dataloader(CONFIG['data_path'], CONFIG['batch_size'], CONFIG['data_fraction'])
    final_results = []
    models_to_evaluate = {
        "Teacher Final": (get_teacher_model, final_teacher_ckpt), "Student Baseline Final": (get_student_model, final_student_base_ckpt), "Student KD Final": (get_student_model, final_student_kd_ckpt)
    }
    for name, (model_fn, ckpt) in models_to_evaluate.items():
        avg_metrics = evaluate_model_on_holdout(name, model_fn, ckpt, holdout_loader); final_results.append(avg_metrics)
    
    print("\n\n" + "="*60 + "\n🎉 RESULTADOS FINALES Y DEFINITIVOS (PROMEDIOS Y MÉTRICAS GLOBALES) 🎉\n" + "="*60)
    display_cols = {"Modelo": "Modelo", "auc_macro_avg": "AUC Macro", "accuracy_macro_avg": "Accuracy Macro", "jaccard_samples_avg": "Jaccard Score (IoU)", "hamming_loss": "Hamming Loss (Error)"}
    final_df = pd.DataFrame(final_results)
    final_df_display = final_df[list(display_cols.keys())].rename(columns=display_cols)
    print(final_df_display.to_string(index=False))


    print("\n\n✅ Proceso completado exitosamente.")

