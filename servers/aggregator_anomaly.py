import flwr as fl
import torch
import argparse
from typing import Dict, List, Tuple
from collections import OrderedDict
from torch.utils.data import DataLoader
import numpy as np
import time
import json
import os

# ✅ IMPORTS CORRETOS
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_anomaly import AnomalyDetectionCNN, SimpleAnomalyDetector
from core.dataset_anomaly import load_anomaly_data
from core.utils_anomaly import get_parameters, set_parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Argumentos EXPANDIDOS ---
parser = argparse.ArgumentParser(description="Agregador Hierárquico")
parser.add_argument("--port", type=int, required=True, help="Porta para o servidor local do agregador.")
parser.add_argument("--global-server-ip", type=str, default="127.0.0.1:8080", help="IP do servidor global")
parser.add_argument("--fog-server-ip", type=str, default="", help="IP do servidor fog (opcional)")
parser.add_argument("--dataset", type=str, default="CIFAR10")
parser.add_argument("--normal-class", type=int, default=0)
parser.add_argument("--num-local-clients", type=int, default=10, help="Número de clientes locais")  # Aumentado
parser.add_argument("--num-local-rounds", type=int, default=4, help="Rodadas locais")
parser.add_argument("--use-transfer-learning", type=bool, default=True)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# --- Configurações do Agregador ---
GLOBAL_SERVER_ADDRESS = f"{args.global_server_ip}:8080"
FOG_SERVER_ADDRESS = f"{args.fog_server_ip}" if args.fog_server_ip else ""
LOCAL_AGGREGATOR_ADDRESS = f"0.0.0.0:{args.port}"
NUM_LOCAL_CLIENTS = args.num_local_clients
NUM_LOCAL_ROUNDS = args.num_local_rounds

torch.manual_seed(args.seed)
final_local_parameters = None
aggregator_metrics = {
    "local_rounds": [],
    "global_communications": [],
    "accuracies": [],
    "client_distribution": {},
    "aggregation_times": []
}

def get_on_fit_config_fn():
    def fit_config(server_round: int):
        return {
            "current_round": server_round,
            "local_epochs": 2,
            "batch_size": 16,
            "server_ip": LOCAL_AGGREGATOR_ADDRESS
        }
    return fit_config

class CustomFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        global final_local_parameters
        start_time = time.time()
        
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None:
            final_local_parameters = aggregated_parameters
            
            # Registra métricas da rodada local
            round_metrics = {
                "round": server_round,
                "clients_participated": len(results),
                "timestamp": time.time(),
                "aggregation_time": time.time() - start_time,
                "client_profiles": [result[1].metrics.get('client_profile', 'unknown') for result in results]
            }
            aggregator_metrics["local_rounds"].append(round_metrics)
            aggregator_metrics["aggregation_times"].append(time.time() - start_time)
            
        return aggregated_parameters, metrics

print(f"🏢 Agregador: Iniciando servidor local em {LOCAL_AGGREGATOR_ADDRESS}")
print(f"👥 Aguardando {NUM_LOCAL_CLIENTS} clientes locais...")
print(f"🔄 Rodadas locais: {NUM_LOCAL_ROUNDS}")
print(f"🎯 Transfer Learning: {args.use_transfer_learning}")
print(f"🌐 Fog Server: {FOG_SERVER_ADDRESS if FOG_SERVER_ADDRESS else 'Não configurado'}")

local_strategy = CustomFedAvg(
    min_available_clients=NUM_LOCAL_CLIENTS,
    min_fit_clients=max(2, NUM_LOCAL_CLIENTS // 2),  # Pelo menos 50% dos clientes
    on_fit_config_fn=get_on_fit_config_fn(),
)

# Servidor local do agregador
fl.server.start_server(
    server_address=LOCAL_AGGREGATOR_ADDRESS,
    config=fl.server.ServerConfig(num_rounds=NUM_LOCAL_ROUNDS),
    strategy=local_strategy,
)

# Decide para qual servidor superior conectar
if FOG_SERVER_ADDRESS:
    UPPER_SERVER_ADDRESS = FOG_SERVER_ADDRESS
    print(f"\n✅ Agregação local finalizada. Conectando ao Fog Server {FOG_SERVER_ADDRESS}...")
else:
    UPPER_SERVER_ADDRESS = GLOBAL_SERVER_ADDRESS
    print(f"\n✅ Agregação local finalizada. Conectando ao Servidor Global {GLOBAL_SERVER_ADDRESS}...")

# Carrega modelo para avaliação
_, test_dataset, ModelClass, model_init_params = load_data(
    args.dataset, num_clients=1, normal_class=args.normal_class,
    use_transfer_learning=args.use_transfer_learning
)
test_loader = DataLoader(test_dataset, batch_size=64)
model = ModelClass(**model_init_params).to(DEVICE)

class AggregatorClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        print(f"🏢 Agregador (porta {args.port}): Enviando modelo agregado para servidor superior.")
        if final_local_parameters is not None:
            return fl.common.parameters_to_ndarrays(final_local_parameters)
        return get_parameters(model)

    def fit(self, parameters, config):
        return self.get_parameters(config), 1, {}

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]):
        start_time = time.time()
        print(f"🏢 Agregador (porta {args.port}): Avaliando modelo superior...")
        
        set_parameters(model, parameters)
        model.eval()

        if isinstance(model, SimpleAnomalyDetector):
            criterion = torch.nn.CrossEntropyLoss()
        else:
            criterion = torch.nn.MSELoss()

        total_loss = 0.0
        correct_anomalies = 0
        total_samples = 0
        all_predictions = []
        all_labels = []
        all_scores = []

        with torch.no_grad():
            for images, labels, is_anomaly in test_loader:
                images, is_anomaly = images.to(DEVICE), is_anomaly.to(DEVICE)
                
                if isinstance(model, SimpleAnomalyDetector):
                    outputs = model(images)
                    loss = criterion(outputs, is_anomaly)
                    _, predicted = torch.max(outputs, 1)
                    scores = torch.softmax(outputs, dim=1)[:, 1]
                    correct_anomalies += (predicted == is_anomaly).sum().item()
                    
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(is_anomaly.cpu().numpy())
                    all_scores.extend(scores.cpu().numpy())
                else:
                    _, reconstructions = model(images)
                    loss = criterion(reconstructions, images)
                    anomaly_scores = torch.mean(torch.pow(reconstructions - images, 2), dim=[1,2,3])
                    predicted_anomalies = (anomaly_scores > 0.1).float()
                    correct_anomalies += (predicted_anomalies == is_anomaly).sum().item()
                    
                    all_predictions.extend(predicted_anomalies.cpu().numpy())
                    all_labels.extend(is_anomaly.cpu().numpy())
                    all_scores.extend(anomaly_scores.cpu().numpy())
                
                total_loss += loss.item() * images.size(0)
                total_samples += images.size(0)

        accuracy = correct_anomalies / total_samples if total_samples > 0 else 0
        average_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        eval_time = time.time() - start_time
        
        from sklearn.metrics import f1_score, roc_auc_score
        f1 = f1_score(all_labels, all_predictions, average='macro')
        try:
            auc = roc_auc_score(all_labels, all_scores)
        except:
            auc = 0.0
        
        # Registra comunicação global
        comm_event = {
            "timestamp": time.time(),
            "parameters_size": sum(p.nbytes for p in parameters),
            "accuracy": accuracy,
            "f1_score": f1,
            "auc": auc,
            "eval_time": eval_time,
            "target_server": "fog" if FOG_SERVER_ADDRESS else "global"
        }
        aggregator_metrics["global_communications"].append(comm_event)
        aggregator_metrics["accuracies"].append(accuracy)
        
        print(f"🏢 Agregador (porta {args.port}): Avaliação - Loss: {average_loss:.4f}, Acc: {accuracy:.4f}, F1: {f1:.4f}, Tempo: {eval_time:.2f}s")
        
        return average_loss, total_samples, {
            "accuracy": accuracy,
            "f1_score": f1,
            "auc": auc,
            "edge_id": args.port
        }

# Conectar ao servidor superior
print(f"🔗 Conectando ao servidor superior {UPPER_SERVER_ADDRESS}...")
fl.client.start_numpy_client(
    server_address=UPPER_SERVER_ADDRESS,
    client=AggregatorClient(),
)

# Salva métricas do agregador
os.makedirs("metrics", exist_ok=True)
metrics_file = f"metrics/aggregator_port_{args.port}_metrics_seed_{args.seed}.json"
with open(metrics_file, "w") as f:
    json.dump(aggregator_metrics, f, indent=2)

print(f"🏢 Agregador (porta {args.port}): Tarefa concluída. Métricas salvas em {metrics_file}")