import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
from skmultilearn.model_selection import IterativeStratification

TARGET_CLASSES_5 = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']
UNCERTAINTY_POLICY = "ones"

class CheXpertDataset(Dataset):
    def __init__(self, dataframe, transform=None, is_train=True):
        self.dataframe = dataframe
        self.transform = transform
        self.labels_df = self.dataframe[TARGET_CLASSES_5]
        # if is_train:
        if UNCERTAINTY_POLICY == "ones":
            self.labels_df = self.labels_df.replace(-1, 1)
        else:
            self.labels_df = self.labels_df.replace(-1, 0)
        self.labels = self.labels_df.values.astype(float)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = os.path.join("/workspace/WORKS/DATA/", self.dataframe.iloc[idx]['Path'])
        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(is_train=False): # Añadir un flag
    if is_train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            # --- NUEVAS TRANSFORMACIONES---
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            # -------------------------
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else: # Para validación y test, no aplicar aumentos aleatorios
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


# def get_transforms():
#     return transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

def create_train_dataloaders(data_path, k_folds, batch_size, data_fraction=1.0):
    train_df_full = pd.read_csv(os.path.join(data_path, 'train.csv'))
    train_df_full['Patient_ID'] = train_df_full['Path'].apply(lambda x: x.split('/')[2])
    train_df_full = train_df_full.fillna(0)

    patient_df = train_df_full.groupby('Patient_ID')[TARGET_CLASSES_5].max().reset_index()


    patient_df.to_csv(os.path.join(data_path, 'patient_df.csv'), index=False) 
    # --- NUEVO: Cargar una fracción de los pacientes para pruebas rápidas ---
    if data_fraction < 1.0:
        print(f"INFO: Usando una fracción del {data_fraction*100:.0f}% de los pacientes para el entrenamiento.")
        patient_df = patient_df.sample(frac=data_fraction, random_state=42)
    
    X_patient = patient_df['Patient_ID'].values.reshape(-1, 1)
    y_patient = patient_df[TARGET_CLASSES_5].values

    k_fold = IterativeStratification(n_splits=k_folds, order=1)
    dataloaders_per_fold = []
    df_per_fold = []

    patient_splits = list(k_fold.split(X_patient, y_patient))
    for train_patient_idx, val_patient_idx in patient_splits:
        train_patients = patient_df.iloc[train_patient_idx]['Patient_ID']
        val_patients = patient_df.iloc[val_patient_idx]['Patient_ID']
        
        train_df = train_df_full[train_df_full['Patient_ID'].isin(train_patients)]
        val_df = train_df_full[train_df_full['Patient_ID'].isin(val_patients)]
        
        df_per_fold.append((train_df, val_df))

        # train_df = train_df.fillna(0)
        # val_df = val_df.fillna(0)

        train_dataset = CheXpertDataset(train_df, transform=get_transforms(is_train=True), is_train=True)
        val_dataset = CheXpertDataset(val_df, transform=get_transforms(is_train=False), is_train=False)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        dataloaders_per_fold.append((train_loader, val_loader))
        
    return dataloaders_per_fold, df_per_fold


def create_final_train_dataloader(data_path, batch_size, data_fraction=1.0):
    """Crea un único dataloader con TODOS los datos de train.csv para el re-entrenamiento final."""
    print("\nINFO: Creando dataloader final con todos los datos de entrenamiento...")
    train_df_full = pd.read_csv(os.path.join(data_path, 'train.csv')).dropna(subset=['Path'])
    
    if data_fraction < 1.0:
        print(f"INFO: Usando una fracción del {data_fraction*100:.0f}% de los datos para el re-entrenamiento final.")
        train_df_full = train_df_full.sample(frac=data_fraction, random_state=42)

    train_df_full = train_df_full.fillna(0)
    
    final_train_dataset = CheXpertDataset(train_df_full, transform=get_transforms(), is_train=True)
    final_train_loader = torch.utils.data.DataLoader(final_train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    return final_train_loader


def get_holdout_validation_dataloader(data_path, batch_size, data_fraction=1.0):
    valid_df = pd.read_csv(os.path.join(data_path, 'valid.csv'))
    
    if data_fraction < 1.0:
        print(f"INFO: Usando una fracción del {data_fraction*100:.0f}% de los datos para la validación final.")
        valid_df = valid_df.sample(frac=data_fraction, random_state=42)

    valid_df = valid_df.fillna(0).replace(-1, 0)
    validation_dataset = CheXpertDataset(valid_df, transform=get_transforms(), is_train=False)
    return torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=2)