import torch
import os
import pandas as pd
import numpy as np
from codecarbon import EmissionsTracker
from tqdm import tqdm

from dataset import create_train_dataloaders, get_holdout_validation_dataloader, create_final_train_dataloader, TARGET_CLASSES_5
from models import get_teacher_model, get_student_model
from trainer import train_one_epoch, validate_one_epoch, save_checkpoint, load_checkpoint
from distillation import DistillationLoss
from utils import plot_distribution_stacked_bar, plot_fold_distribution_stacked_bar, plot_learning_curves, plot_metric_learning_curves, plot_roc_curves, calculate_metrics
from dataset import CheXpertDataset , get_transforms
import json  


CONFIG = {
    "data_path": "/workspace/WORKS/DATA/CheXpert-v1.0-small",
    "k_folds": 3,
    "num_epochs": 30,
    "batch_size": 64,
    "learning_rate": 0.001,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "results_dir": "results_19062025_01",
    "data_fraction": 1.0, 
    "alpha": 0.3,
    "temperature": 5.0,
}

def print_config(config_dict):
    print("\n" + "="*50 + "\n⚙️  PARÁMETROS DE LA EJECUCIÓN  ⚙️\n" + "="*50)
    for key, value in config_dict.items(): print(f"{key:<20}: {value}")
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

def run_kfold_experiment(model_name, model, train_loader, val_loader, fold_idx, is_distillation=False, teacher_model=None):
    print(f"\n--- Entrenando: {model_name} | Fold: {fold_idx+1}/{CONFIG['k_folds']} ---")
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    checkpoint_path = os.path.join(CONFIG['results_dir'], f"{model_name}_fold_{fold_idx}_checkpoint.pth")
    best_val_auc = 0.0
    start_epoch, best_val_auc, history = load_checkpoint(checkpoint_path, model, optimizer)
    history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}
    
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
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, CONFIG['device'], train_desc, loss_fn_kd)
        val_loss, val_metrics = validate_one_epoch(model, val_loader, criterion, CONFIG['device'], TARGET_CLASSES_5, val_desc)
        
        history['train_loss'].append(train_loss) # Se guarda el train_loss real
        history['val_loss'].append(val_loss)     # Se guarda el val_loss real
        history['val_metrics'].append(val_metrics)
        
        current_auc = val_metrics['auc_macro_avg']
        
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC (Macro Avg): {current_auc:.4f}")

        if current_auc > best_val_auc:
            best_val_auc = current_auc
            save_checkpoint(epoch, model, optimizer, best_val_auc, history, checkpoint_path)
            print(f"✅ Nuevo mejor checkpoint guardado para {model_name} (Fold {fold_idx+1}) con AUC: {best_val_auc:.4f}")
    
    
    
    print(f"\n📊 Generando gráficas de aprendizaje para {model_name} (Fold {fold_idx+1})...")
    plot_learning_curves(history, fold_idx, model_name, CONFIG['results_dir'])
    
    for metric in ["AUC", "F1-Score", "Precision", "Recall", "Accuracy"]:
        plot_metric_learning_curves(history, TARGET_CLASSES_5, metric, fold_idx, model_name, CONFIG['results_dir'])

    return max(history['val_metrics'], key=lambda x: x['auc_macro_avg'])

def train_final_model(model_name, model, final_train_loader, is_distillation=False, teacher_model=None):
    print(f"\n{'='*20} Re-entrenamiento Final: {model_name} {'='*20}")
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    final_checkpoint_path = os.path.join(CONFIG['results_dir'], f"{model_name}_final_model.pth")
    
    model.to(CONFIG['device'])
    loss_fn_kd = None
    if is_distillation and teacher_model:
        print("INFO: Usando destilación para el re-entrenamiento final.")
        teacher_model.to(CONFIG['device']).eval()
        loss_fn_kd = DistillationLoss(criterion, teacher_model, CONFIG['alpha'], CONFIG['temperature'])
    
    for epoch in range(CONFIG['num_epochs']):
        train_desc = f"FINAL Train {model_name} [Epoch {epoch+1}/{CONFIG['num_epochs']}]"
        train_one_epoch(model, final_train_loader, criterion, optimizer, CONFIG['device'], train_desc, loss_fn_kd)
        print(f"Epoch {epoch+1} de re-entrenamiento final completada para {model_name}.")

    torch.save(model.state_dict(), final_checkpoint_path)
    print(f"✅ Modelo final guardado en: {final_checkpoint_path}")
    return final_checkpoint_path

def evaluate_model_on_holdout(model_name, model_class, checkpoint_path, holdout_loader):
    print(f"\n--- 🧪 Evaluando {model_name} en el Hold-Out Set (`valid.csv`) ---")
    model = model_class(num_classes=len(TARGET_CLASSES_5))
    model.load_state_dict(torch.load(checkpoint_path, map_location=CONFIG['device']))
    model.to(CONFIG['device']).eval()
    
    all_labels, all_preds = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(holdout_loader, desc=f"Evaluando {model_name}"):
            outputs = model(inputs.to(CONFIG['device']))
            all_labels.append(labels)
            all_preds.append(torch.sigmoid(outputs.cpu()))
    
    all_labels, all_preds = torch.cat(all_labels), torch.cat(all_preds)
    metrics = calculate_metrics(all_labels, all_preds, TARGET_CLASSES_5)
    plot_roc_curves(all_labels, all_preds, TARGET_CLASSES_5, model_name, CONFIG['results_dir'])
    
    print(f"\n--- Métricas Detalladas para: {model_name} ---")
    header = f"{'Patología':<18} | {'AUC':<6} | {'F1':<6} | {'Precision':<9} | {'Recall':<6}"
    print(header); print('-' * len(header.replace('|', '+')))
    for pathology in TARGET_CLASSES_5:
        auc, f1, precision, recall = [metrics.get(f'{m}_{pathology}', 0) for m in ['auc', 'f1', 'precision', 'recall']]
        print(f"{pathology:<18} | {auc:.4f} | {f1:.4f} | {precision:.4f}  | {recall:.4f}")
    
    return {"Modelo": model_name, **metrics}


if __name__ == '__main__':
    os.makedirs(CONFIG['results_dir'], exist_ok=True)
    tracker = EmissionsTracker(output_dir=CONFIG['results_dir'], project_name="Final_CheXpert_KD")
    tracker.start()
    print_config(CONFIG)

    # FASE 1: VALIDACIÓN CRUZADA
    print("\n" + "#"*60 + "\n# FASE 1: INICIANDO VALIDACIÓN CRUZADA (K-FOLD) PARA ESTIMACIÓN\n" + "#"*60)
    dataloaders_kfold = create_train_dataloaders(CONFIG['data_path'], CONFIG['k_folds'], CONFIG['batch_size'], CONFIG['data_fraction'])
    all_fold_results = {"Teacher": [], "Student_Baseline": [], "Student_KD": []}

    train_df = pd.read_csv(os.path.join(CONFIG['data_path'], 'train.csv'))
    val_df = pd.read_csv(os.path.join(CONFIG['data_path'], 'valid.csv'))
   
    train_dataset = CheXpertDataset(train_df, transform=get_transforms(), is_train=True)
    val_dataset = CheXpertDataset(val_df, transform=get_transforms(), is_train=True)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)
       

    plot_distribution_stacked_bar(
            train_df=train_loader.dataset.dataframe,
            val_df=val_loader.dataset.dataframe,
            class_names=TARGET_CLASSES_5,
            results_dir=CONFIG['results_dir']
        )
    
    
    
    for fold_idx, (train_loader, val_loader) in enumerate(dataloaders_kfold):
        print(f"\n{'='*25} INICIANDO FOLD {fold_idx + 1}/{CONFIG['k_folds']} {'='*25}")
        

        plot_fold_distribution_stacked_bar(
            train_df=train_loader.dataset.dataframe,
            val_df=val_loader.dataset.dataframe,
            class_names=TARGET_CLASSES_5,
            fold_idx=fold_idx,
            results_dir=CONFIG['results_dir']
        )
        
        models_to_run_in_fold = ["Teacher", "Student_Baseline", "Student_KD"]
        fold_metrics_data = {}
        teacher_model_for_distill = get_teacher_model(num_classes=len(TARGET_CLASSES_5)) # Para cargarlo después

        for model_name in models_to_run_in_fold:
            metrics_path = os.path.join(CONFIG['results_dir'], f"{model_name}_fold_{fold_idx}_metrics.json")
            checkpoint_path = os.path.join(CONFIG['results_dir'], f"{model_name}_fold_{fold_idx}_checkpoint.pth") # .pth para reanudar y para cargar

            if os.path.exists(metrics_path):
                print(f"INFO: Cargando resultados existentes para {model_name} (Fold {fold_idx+1}). Saltando entrenamiento.")
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
            else:
                if model_name == "Teacher":
                    model = get_teacher_model(num_classes=len(TARGET_CLASSES_5))
                    metrics = run_kfold_experiment(model_name, model, train_loader, val_loader, fold_idx)
                elif model_name == "Student_Baseline":
                    model = get_student_model(num_classes=len(TARGET_CLASSES_5))
                    metrics = run_kfold_experiment(model_name, model, train_loader, val_loader, fold_idx)
                elif model_name == "Student_KD":
                    model = get_student_model(num_classes=len(TARGET_CLASSES_5))
                    # Cargar el teacher de este fold para la destilación
                    teacher_model_for_distill.load_state_dict(torch.load(os.path.join(CONFIG['results_dir'], f"Teacher_fold_{fold_idx}_checkpoint.pth"))['model_state_dict'])
                    metrics = run_kfold_experiment(model_name, model, train_loader, val_loader, fold_idx, is_distillation=True, teacher_model=teacher_model_for_distill)
                
                # Guardar el "recibo" de métricas al completar
                if metrics: # Asegurarse de que el entrenamiento no devolvió un diccionario vacío
                    with open(metrics_path, 'w') as f:
                        json.dump(metrics, f, indent=4)
            
            if metrics:
                all_fold_results[model_name].append(metrics)
                fold_metrics_data[model_name] = metrics


        print(f"\n--- Resumen de Rendimiento para el Fold {fold_idx + 1} (Mejores Métricas de la Época) ---")
        fold_summary_list = [{"Modelo": name, **data} for name, data in fold_metrics_data.items()]
        fold_df = pd.DataFrame(fold_summary_list)
        display_cols = {"Modelo": "Modelo", "accuracy_macro_avg": "Accuracy", "auc_macro_avg": "AUC", "f1_macro_avg": "F1-Score"}
        fold_df_display = fold_df[list(display_cols.keys())].rename(columns=display_cols)
        print(fold_df_display.to_string(index=False))





    print("\n\n" + "="*50 + "\n📊 RESUMEN FINAL DE VALIDACIÓN CRUZADA (K-FOLD) 📊\n" + "="*50)
    summary_data = []
    for model_name, fold_metrics_list in all_fold_results.items():
        aucs = [m['auc_macro_avg'] for m in fold_metrics_list]
        accuracies = [m['accuracy_macro_avg'] for m in fold_metrics_list]
        f1_scores = [m['f1_macro_avg'] for m in fold_metrics_list]
        
        summary_data.append({
            "Modelo": model_name,
            "AUC Media": f"{np.mean(aucs):.4f}",
            "AUC Std Dev": f"{np.std(aucs):.4f}",
            "Accuracy Media": f"{np.mean(accuracies):.4f}",
            "Accuracy Std Dev": f"{np.std(accuracies):.4f}",
            "F1 Media": f"{np.mean(f1_scores):.4f}"
        })
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    print("\n\n" + "#"*60 + "\n# FASE 2: INICIANDO RE-ENTRENAMIENTO FINAL CON TODOS LOS DATOS\n" + "#"*60)
    final_train_loader = create_final_train_dataloader(CONFIG['data_path'], CONFIG['batch_size'], CONFIG['data_fraction'])
    final_teacher_model = get_teacher_model(num_classes=len(TARGET_CLASSES_5)); final_teacher_ckpt = train_final_model("Teacher", final_teacher_model, final_train_loader)
    final_teacher_model.load_state_dict(torch.load(final_teacher_ckpt))
    final_student_base_model = get_student_model(num_classes=len(TARGET_CLASSES_5)); final_student_base_ckpt = train_final_model("Student_Baseline", final_student_base_model, final_train_loader)
    final_student_kd_model = get_student_model(num_classes=len(TARGET_CLASSES_5)); final_student_kd_ckpt = train_final_model("Student_KD", final_student_kd_model, final_train_loader, is_distillation=True, teacher_model=final_teacher_model)
    
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
    tracker.stop()
    report_cumulative_emissions(os.path.join(CONFIG['results_dir'], 'emissions.csv'))
    print("\n\n✅ Proceso completado exitosamente.")
