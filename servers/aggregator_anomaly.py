import flwr as fl
import torch
import argparse
from typing import Dict, List
from collections import OrderedDict
from torch.utils.data import DataLoader
import numpy as np
import time
import json
import os
import sys
import psutil

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.model_anomaly import AnomalyDetectionCNN, SimpleAnomalyDetector
from core.dataset_anomaly import load_anomaly_data
from core.utils_anomaly import get_parameters, set_parameters

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Argumentos
parser = argparse.ArgumentParser(description="Agregador Hierárquico")
parser.add_argument("--port", type=int, required=True, help="Porta do agregador")
parser.add_argument("--server-ip", type=str, default="127.0.0.1:8080", help="IP do servidor (Fog)")
parser.add_argument("--min-clients", type=int, default=2, help="Mínimo de clientes")
parser.add_argument("--dataset", type=str, default="CIFAR10")
parser.add_argument("--use-transfer-learning", type=bool, default=True)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

torch.manual_seed(args.seed)
aggregator_address = f"0.0.0.0:{args.port}"
aggregator_parameters = None 

aggregator_metrics = {
    "local_rounds": [],
    "communications": [],
    "accuracies": []
}

class CustomFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        global aggregator_parameters
        start_time = time.time()
        
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None:
            aggregator_parameters = aggregated_parameters
            
            round_metrics = {
                "round": server_round,
                "clients_participated": len(results),
                "aggregation_time": time.time() - start_time,
            }
            aggregator_metrics["local_rounds"].append(round_metrics)
            
        return aggregated_parameters, metrics

# ✅ CORREÇÃO: Carrega APENAS o dataset de teste (load_train=False)
# Isso evita o processamento lento das 50.000 imagens de treino.
print("... Carregando dataset de teste (rápido)...")
_, test_dataset, ModelClass, model_config = load_anomaly_data(
    args.dataset, num_clients=1, normal_class=0,
    use_transfer_learning=args.use_transfer_learning,
    load_train=False, load_test=True 
)
test_loader = DataLoader(test_dataset, batch_size=32)
model = ModelClass(**model_config).to(DEVICE)

class AggregatorClient(fl.client.NumPyClient):
    
    def __init__(self):
        self.model = ModelClass(**model_config).to(DEVICE)
        self.parameters = get_parameters(self.model)

    def get_parameters(self, config):
        print(f"🏢 Agregador: Enviando modelo para servidor (Fog)")
        return self.parameters

    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]):
        global aggregator_parameters
        
        print(f"🏢 Agregador [FIT]: Recebeu modelo do Fog. Iniciando rodada local para Clientes...")
        set_parameters(self.model, parameters)
        
        strategy = CustomFedAvg(
            fraction_fit=1.0,
            min_fit_clients=args.min_clients,
            min_available_clients=args.min_clients,
            fraction_evaluate=0.0,
            min_evaluate_clients=args.min_clients,
            initial_parameters=fl.common.ndarrays_to_parameters(parameters) 
        )
        
        # Inicia um servidor de 1 rodada para os Clientes
        fl.server.start_server(
            server_address=aggregator_address,
            config=fl.server.ServerConfig(num_rounds=1),
            strategy=strategy,
            server=fl.server.Server(client_manager=fl.server.SimpleClientManager(), strategy=strategy)
        )
        
        print(f"🏢 Agregador [FIT]: Agregação local concluída.")
        
        if aggregator_parameters is not None:
            self.parameters = fl.common.parameters_to_ndarrays(aggregator_parameters)
            return self.parameters, len(test_loader.dataset), {}
        else:
            return self.parameters, len(test_loader.dataset), {}

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]):
        start_time = time.time()
        print(f"🏢 Agregador: Avaliando modelo...")
        
        set_parameters(model, parameters)
        model.eval()

        if isinstance(model, SimpleAnomalyDetector):
            criterion = torch.nn.CrossEntropyLoss()
        else:
            criterion = torch.nn.MSELoss()

        total_loss = 0.0
        correct_anomalies = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels, is_anomaly in test_loader:
                images, is_anomaly = images.to(DEVICE), is_anomaly.to(DEVICE)
                
                if isinstance(model, SimpleAnomalyDetector):
                    outputs = model(images)
                    loss = criterion(outputs, is_anomaly)
                    _, predicted = torch.max(outputs, 1)
                    correct_anomalies += (predicted == is_anomaly).sum().item()
                else:
                    _, reconstructions = model(images)
                    loss = criterion(reconstructions, images)
                    anomaly_scores = torch.mean(torch.pow(reconstructions - images, 2), dim=[1,2,3])
                    predicted_anomalies = (anomaly_scores > 0.1).float()
                    correct_anomalies += (predicted_anomalies == is_anomaly).sum().item()
                
                total_loss += loss.item() * images.size(0)
                total_samples += images.size(0)

        accuracy = correct_anomalies / total_samples if total_samples > 0 else 0
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        eval_time = time.time() - start_time
        
        comm_event = {"accuracy": accuracy, "eval_time": eval_time}
        aggregator_metrics["communications"].append(comm_event)
        aggregator_metrics["accuracies"].append(accuracy)
        
        print(f"🏢 Agregador: Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}")
        
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        print(f"📊 Agregador: Uso de memória: {mem_info.rss / (1024 * 1024):.2f} MB (RSS), {mem_info.vms / (1024 * 1024):.2f} MB (VMS)")
        return avg_loss, total_samples, {"accuracy": accuracy}

try:
    print(f"🏢 Agregador: Conectando ao Fog em {args.server_ip}...")
    fl.client.start_client(
        server_address=args.server_ip,
        client=AggregatorClient().to_client(),
    )
    
    os.makedirs("results", exist_ok=True)
    metrics_file = f"results/aggregator_port_{args.port}_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(aggregator_metrics, f, indent=2)
    print(f"💾 Métricas salvas: {metrics_file}")
    
except Exception as e:
    print(f"❌ Erro ao conectar com servidor Fog: {e}")