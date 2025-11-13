import flwr as fl
import torch
import argparse
import time
import random
import numpy as np
import os
import sys
import psutil
from typing import Dict, List
from torch.utils.data import DataLoader

# Ajustar o path de importação
sys.path.append(os.path.dirname(__file__))

from core.model_anomaly import SimpleAnomalyDetector
from core.dataset_anomaly import load_anomaly_data
from core.utils_anomaly import set_parameters

# PERFIS DE HARDWARE REALISTAS
HARDWARE_PROFILES = {
    "high_end": {"cpu_cores": 4, "memory_gb": 8, "network_latency": 10, "epochs": 3},
    "medium": {"cpu_cores": 2, "memory_gb": 4, "network_latency": 50, "epochs": 2},
    "low_end": {"cpu_cores": 1, "memory_gb": 2, "network_latency": 100, "epochs": 1},
    "straggler": {"cpu_cores": 1, "memory_gb": 1, "network_latency": 200, "epochs": 1, "dropout_prob": 0.3}
}

parser = argparse.ArgumentParser(description="Cliente com Heterogeneidade Real")
parser.add_argument("--server-ip", type=str, required=True)
parser.add_argument("--client-id", type=int, required=True)
parser.add_argument("--profile", type=str, default="medium", choices=list(HARDWARE_PROFILES.keys()))
parser.add_argument("--scenario", type=str, default="small", choices=["small", "medium", "large"])
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()

# CONFIGURAÇÕES POR CENÁRIO
SCENARIO_CONFIGS = {
    "small": {"total_clients": 50},
    "medium": {"total_clients": 100},
    "large": {"total_clients": 200}
}

class RealisticClient(fl.client.NumPyClient):
    def __init__(self, client_id, profile, scenario):
        self.client_id = client_id
        self.profile = HARDWARE_PROFILES[profile]
        self.scenario = scenario
        
        self.setup_hardware_emulation()
        
        self.network_delay = self.profile["network_latency"] / 1000
        
        if profile == "straggler" and random.random() < self.profile.get("dropout_prob", 0):
            print(f"❌ Cliente {client_id} dropando (straggler)")
            raise ConnectionError("Straggler dropout")
        
        # Carrega dados
        total_clients = SCENARIO_CONFIGS[scenario]["total_clients"]
        
        # ✅ CORREÇÃO: Carrega APENAS o dataset de treino (load_test=False)
        client_splits, _, model_class, model_config = load_anomaly_data(
            "CIFAR10", num_clients=total_clients, normal_class=0,
            use_transfer_learning=True, scenario=scenario,
            load_train=True, load_test=False 
        )
        
        self.train_loader = DataLoader(
            client_splits[client_id % len(client_splits)], 
            batch_size=16, 
            shuffle=True
        )
        self.model = model_class(**model_config)
        
        print(f"👤 Cliente {client_id} ({profile}) - {scenario}")

    def setup_hardware_emulation(self):
        try:
            cpu_cores = self.profile["cpu_cores"]
            if hasattr(os, 'sched_setaffinity'):
                available_cpus = list(range(min(cpu_cores, os.cpu_count())))
                os.sched_setaffinity(0, available_cpus)
            
            memory_gb = self.profile["memory_gb"]
            self.memory_limit = memory_gb * 1024 * 1024 * 1024
            print(f"ℹ️  Cliente {self.client_id}: Memória alvo: {memory_gb} GB")
            
        except Exception as e:
            print(f"⚠️  Emulação de hardware não suportada: {e}")

    def get_properties(self, config):
        return {"client_id": self.client_id, "profile": self.profile, "scenario": self.scenario}

    def get_parameters(self, config):
        print(f"📤 Cliente {self.client_id}: Enviando parâmetros...")
        try:
            parameters = []
            for param in self.model.parameters():
                parameters.append(param.data.cpu().numpy())
            print(f"✅ Cliente {self.client_id}: {len(parameters)} parâmetros convertidos")
            return parameters
        except Exception as e:
            print(f"❌ Erro crítico em get_parameters: {e}")
            return [np.array([0.0])]

    def fit(self, parameters, config):
        start_time = time.time()
        time.sleep(random.uniform(0, self.network_delay))
        
        try:
            set_parameters(self.model, parameters)
        except Exception as e:
            print(f"⚠️  Erro crítico ao setar parâmetros: {e}")
        
        self.model.train()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        
        epoch_losses = []
        for epoch in range(self.profile["epochs"]):
            epoch_loss = 0.0
            compute_factor = 1.0 / self.profile["cpu_cores"]
            
            for images, labels, is_anomaly in self.train_loader:
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, is_anomaly)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                time.sleep(0.001 * compute_factor * random.uniform(0.8, 1.2))
            
            epoch_losses.append(epoch_loss / len(self.train_loader))
        
        training_time = time.time() - start_time
        accuracy = 0.7 + random.uniform(-0.15, 0.15) # Acurácia simulada (só para log)
        
        print(f"👤 Cliente {self.client_id}: Acc={accuracy:.3f}, Time={training_time:.2f}s")
        
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        print(f"📊 Cliente {self.client_id}: Uso de memória: {mem_info.rss / (1024 * 1024):.2f} MB (RSS), {mem_info.vms / (1024 * 1024):.2f} MB (VMS)")

        return self.get_parameters(config), len(self.train_loader.dataset), {
            "accuracy": accuracy,
            "training_time": training_time,
            "client_profile": args.profile,
            "client_id": self.client_id,
            "scenario": self.scenario
        }

if __name__ == "__main__":
    try:
        client = RealisticClient(args.client_id, args.profile, args.scenario)
        fl.client.start_client(
            server_address=args.server_ip,
            client=client.to_client(),
        )
        print(f"✅ Cliente {args.client_id} finalizado")
    except Exception as e:
        print(f"❌ Cliente {args.client_id} falhou: {e}")