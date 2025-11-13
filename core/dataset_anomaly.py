import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
import numpy as np
import os

class AnomalyDataset(Dataset):
    def __init__(self, normal_images, anomaly_images=None, transform=None):
        self.normal_images = normal_images
        self.anomaly_images = anomaly_images if anomaly_images is not None else []
        self.transform = transform
        
        self.labels = [0] * len(normal_images) + [1] * len(anomaly_images)
        self.images = normal_images + anomaly_images
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        is_anomaly = self.labels[idx]
        
        if self.transform and not isinstance(image, torch.Tensor):
            image = self.transform(image)
        elif isinstance(image, torch.Tensor) and self.transform:
            # Re-aplicar transform se já for tensor (para augmentation em cache)
            image = self.transform(image)
            
        return image, torch.tensor(label), torch.tensor(is_anomaly)

# ✅ CORREÇÃO: Adicionado load_train e load_test
def load_anomaly_data(dataset_name, num_clients, normal_class=0, anomaly_ratio=0.1, 
                     use_augmentation=True, use_transfer_learning=True, scenario="medium",
                     load_train=True, load_test=True):
    """Carrega dados para detecção de anomalias"""
    
    scenario_configs = {
        "small": {"anomaly_ratio": 0.15, "data_ratio": 0.6},
        "medium": {"anomaly_ratio": 0.1, "data_ratio": 0.8}, 
        "large": {"anomaly_ratio": 0.08, "data_ratio": 1.0}
    }
    
    config = scenario_configs.get(scenario, scenario_configs["medium"])
    anomaly_ratio = config["anomaly_ratio"]
    data_ratio = config["data_ratio"]
    
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # ✅ CORREÇÃO: Inicializa variáveis
    client_splits = None
    test_dataset = None
    normal_images_count = 0
    anomaly_images_count = 0

    if dataset_name == "CIFAR10":
        
        # ✅ CORREÇÃO: Só processa o TREINO se for pedido
        if load_train:
            train_dataset_raw = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
            
            total_train_size = len(train_dataset_raw)
            used_train_size = int(total_train_size * data_ratio)
            train_dataset, _ = random_split(train_dataset_raw, [used_train_size, total_train_size - used_train_size])
            
            normal_images = []
            anomaly_images = []
            
            print("... Processando dataset de treino (pode demorar)...")
            for img, label in train_dataset: # Esta é a parte lenta
                if label == normal_class:
                    normal_images.append(img)
                else:
                    anomaly_images.append(img)
            
            num_anomalies = int(len(normal_images) * anomaly_ratio)
            if len(anomaly_images) > num_anomalies:
                anomaly_images = anomaly_images[:num_anomalies]
            
            normal_images_count = len(normal_images)
            anomaly_images_count = len(anomaly_images)
            
            full_train_dataset = AnomalyDataset(normal_images, anomaly_images, transform=None)
            
            total_len = len(full_train_dataset)
            len_per_client = total_len // num_clients
            remainder = total_len % num_clients
            lengths = [len_per_client] * num_clients
            for i in range(remainder):
                lengths[i] += 1
                
            client_splits = random_split(full_train_dataset, lengths, generator=torch.Generator().manual_seed(42))
        
        # ✅ CORREÇÃO: Só processa o TESTE se for pedido
        if load_test:
            test_dataset_raw = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)
            
            test_normal = []
            test_anomaly = []
            
            for img, label in test_dataset_raw:
                if label == normal_class:
                    test_normal.append(img)
                else:
                    test_anomaly.append(img)
                    
            test_anomaly = test_anomaly[:int(len(test_normal) * anomaly_ratio)]
            test_dataset = AnomalyDataset(test_normal, test_anomaly, transform=None)
            
    else:
        raise ValueError(f"Dataset {dataset_name} não suportado")
    
    print(f"📊 Dataset {dataset_name} - Cenário {scenario}:")
    if load_train:
        print(f"   👥 {normal_images_count} normais, {anomaly_images_count} anomalias (Treino)")
    
    if use_transfer_learning:
        from core.model_anomaly import SimpleAnomalyDetector
        model_class = SimpleAnomalyDetector
        model_config = {"in_channels": 3, "num_classes": 2}
    else:
        from core.model_anomaly import AnomalyDetectionCNN
        model_class = AnomalyDetectionCNN
        model_config = {"in_channels": 3, "latent_dim": 128, "image_size": 64}
    
    return client_splits, test_dataset, model_class, model_config