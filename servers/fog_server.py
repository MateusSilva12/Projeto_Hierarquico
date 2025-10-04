import flwr as fl
import torch
import argparse
from typing import Dict, List, Tuple
from collections import OrderedDict
import numpy as np
import time
import json
import os

from core.model_anomaly import AnomalyDetectionCNN, SimpleAnomalyDetector
from core.dataset_anomaly import load_anomaly_data
from core.utils_anomaly import get_parameters, set_parameters

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configurações do Fog Server
parser = argparse.ArgumentParser(description="Fog Server - Camada Intermediária")
parser.add_argument("--fog-port", type=int, required=True, help="Porta do servidor fog")
parser.add_argument("--cloud-ip", type=str, default="127.0.0.1:8080", help="IP do servidor cloud")
parser.add_argument("--edge-ports", type=str, required=True, help="Portas dos edges separadas por vírgula")
parser.add_argument("--dataset", type=str, default="CIFAR10")
parser.add_argument("--num-fog-rounds", type=int, default=3, help="Rodadas de agregação no fog")
parser.add_argument("--use-transfer-learning", type=bool, default=True)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# Processa portas dos edges
edge_ports = [int(port.strip()) for port in args.edge_ports.split(',')]
EDGE_SERVERS = [f"127.0.0.1:{port}" for port in edge_ports]
FOG_SERVER_ADDRESS = f"0.0.0.0:{args.fog_port}"

torch.manual_seed(args.seed)
fog_metrics = {
    "fog_rounds": [],
    "cloud_communications": [],
    "aggregation_times": [],
    "edge_connections": []
}

class FogFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.edge_models = {}
    
    def aggregate_fit(self, server_round, results, failures):
        start_time = time.time()
        
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None:
            # Registra métricas da rodada fog
            round_metrics = {
                "round": server_round,
                "edges_participated": len(results),
                "aggregation_time": time.time() - start_time,
                "timestamp": time.time()
            }
            fog_metrics["fog_rounds"].append(round_metrics)
            
            # Armazena modelos dos edges
            for result in results:
                edge_id = result[1].metrics.get('edge_id', 'unknown')
                self.edge_models[edge_id] = result[0]
        
        return aggregated_parameters, metrics

print(f"🌫️  Fog Server: Iniciando em {FOG_SERVER_ADDRESS}")
print(f"🔗 Conectando a {len(EDGE_SERVERS)} edges: {EDGE_SERVERS}")
print(f"🔄 Rodadas Fog: {args.num_fog_rounds}")

fog_strategy = FogFedAvg(
    min_available_clients=len(EDGE_SERVERS),
    min_fit_clients=len(EDGE_SERVERS),
    fraction_fit=1.0
)

# Servidor Fog
fl.server.start_server(
    server_address=FOG_SERVER_ADDRESS,
    config=fl.server.ServerConfig(num_rounds=args.num_fog_rounds),
    strategy=fog_strategy,
)

print(f"\n✅ Agregação Fog finalizada. Conectando ao Cloud {args.cloud_ip}...")

# Prepara modelo para avaliação
_, test_dataset, ModelClass, model_init_params = load_anomaly_data(
    args.dataset, num_clients=1, normal_class=0,
    use_transfer_learning=args.use_transfer_learning
)

model = ModelClass(**model_init_params).to(DEVICE)

class FogClient(fl.client.NumPyClient):
    def __init__(self):
        self.final_fog_parameters = None
    
    def get_parameters(self, config):
        print(f"🌫️  Fog (porta {args.fog_port}): Enviando modelo agregado para o cloud.")
        if self.final_fog_parameters is not None:
            return self.final_fog_parameters
        return get_parameters(model)
    
    def set_final_parameters(self, parameters):
        self.final_fog_parameters = parameters
    
    def fit(self, parameters, config):
        # Fog não treina localmente, apenas agrega
        return self.get_parameters(config), 1, {}
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]):
        start_time = time.time()
        print(f"🌫️  Fog (porta {args.fog_port}): Avaliando modelo cloud...")
        
        set_parameters(model, parameters)
        model.eval()
        
        if isinstance(model, SimpleAnomalyDetector):
            criterion = torch.nn.CrossEntropyLoss()
        else:
            criterion = torch.nn.MSELoss()
        
        # Avaliação simplificada no fog
        from torch.utils.data import DataLoader
        test_loader = DataLoader(test_dataset, batch_size=32)
        
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
        average_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        eval_time = time.time() - start_time
        
        # Registra comunicação cloud
        comm_event = {
            "timestamp": time.time(),
            "parameters_size": sum(p.nbytes for p in parameters),
            "accuracy": accuracy,
            "eval_time": eval_time
        }
        fog_metrics["cloud_communications"].append(comm_event)
        
        print(f"🌫️  Fog (porta {args.fog_port}): Avaliação - Loss: {average_loss:.4f}, Acc: {accuracy:.4f}")
        
        return average_loss, total_samples, {"accuracy": accuracy}

# Conecta ao Cloud
print(f"🔗 Conectando ao Cloud {args.cloud_ip}...")
fog_client = FogClient()

# Obtém parâmetros finais do fog (última agregação)
if hasattr(fog_strategy, 'edge_models') and fog_strategy.edge_models:
    # Simula agregação final dos modelos dos edges
    all_parameters = list(fog_strategy.edge_models.values())
    if all_parameters:
        # Média simples dos modelos dos edges
        aggregated_params = []
        for param_list in zip(*all_parameters):
            aggregated_params.append(np.mean(param_list, axis=0))
        fog_client.set_final_parameters(aggregated_params)

fl.client.start_numpy_client(
    server_address=args.cloud_ip,
    client=fog_client,
)

# Salva métricas do fog
metrics_file = f"metrics/fog_port_{args.fog_port}_metrics_seed_{args.seed}.json"
os.makedirs("metrics", exist_ok=True)
with open(metrics_file, "w") as f:
    json.dump(fog_metrics, f, indent=2)

print(f"🌫️  Fog Server (porta {args.fog_port}): Tarefa concluída. Métricas salvas em {metrics_file}")