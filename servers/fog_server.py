import flwr as fl
import torch
import argparse
import time
import json
import os
import sys
from typing import Dict, List, Tuple, Union
from flwr.common import Metrics
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.model_anomaly import SimpleAnomalyDetector
from core.dataset_anomaly import load_anomaly_data
from core.utils_anomaly import get_parameters, set_parameters

parser = argparse.ArgumentParser(description="Fog Server - Camada Intermediária")
parser.add_argument("--fog-port", type=int, required=True)
parser.add_argument("--cloud-ip", type=str, required=True)
parser.add_argument("--edge-ports", type=str, required=True) 
parser.add_argument("--scenario", type=str, default="small", choices=["small", "medium", "large"])
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()

edge_ports = [int(port.strip()) for port in args.edge_ports.split(',')]
EDGE_SERVERS = [f"127.0.0.1:{port}" for port in edge_ports]
FOG_SERVER_ADDRESS = f"0.0.0.0:{args.fog_port}"

print(f"🌫️  FOG SERVER: Porta {args.fog_port}")
print(f"🔗 Conectando a {len(EDGE_SERVERS)} edges")
print(f"📊 Cenário: {args.scenario}")

SCENARIO_CONFIGS = {
    "small": {"fog_rounds": 3, "min_edges": 1},
    "medium": {"fog_rounds": 4, "min_edges": 2},
    "large": {"fog_rounds": 5, "min_edges": 3}
}

fog_config = SCENARIO_CONFIGS[args.scenario]

# ✅ CORREÇÃO: Variáveis globais para armazenar o último modelo e acurácia
fog_parameters = None
fog_real_accuracy = 0.0
fog_real_loss = 0.0

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Agrega acurácias reais recebidas dos Agregadores."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics if "accuracy" in m]
    examples = [num_examples for num_examples, m in metrics if "accuracy" in m]
    
    if not examples:
        return {}
        
    avg_accuracy = sum(accuracies) / sum(examples)
    print(f"🌫️  Fog: Média real dos Agregadores: {avg_accuracy:.4f}")
    return {"accuracy": avg_accuracy}

class FogFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aggregation_times = []
    
    def aggregate_fit(self, server_round, results, failures):
        global fog_parameters
        start_time = time.time()
        
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None:
            fog_parameters = aggregated_parameters # Salva o modelo agregado
            agg_time = time.time() - start_time
            self.aggregation_times.append(agg_time)
            print(f"🌫️  Fog - Rodada {server_round}: {len(results)} edges, {agg_time:.2f}s")
        
        return aggregated_parameters, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        """Agrega métricas reais e as salva globalmente."""
        global fog_real_accuracy, fog_real_loss
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        
        if metrics and "accuracy" in metrics:
            fog_real_accuracy = metrics["accuracy"]
            fog_real_loss = loss if loss is not None else 0.0
        
        return loss, metrics

# 🛑 CORREÇÃO: REMOVIDO o start_server() inicial.

class FogClient(fl.client.NumPyClient):
    
    def __init__(self):
        # ✅ CORREÇÃO: O Fog precisa manter seu próprio modelo
        self.model = SimpleAnomalyDetector(in_channels=3, num_classes=2)
        self.parameters = get_parameters(self.model)

    def get_parameters(self, config):
        print(f"🌫️  Fog: Enviando parâmetros para Cloud")
        return self.parameters
    
    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]):
        global fog_parameters
        
        print(f"🌫️  Fog [FIT]: Recebeu modelo do Cloud. Iniciando rodada local para Agregadores...")
        # 1. Atualiza o modelo local com os parâmetros do Cloud
        set_parameters(self.model, parameters)
        
        # 2. Define a estratégia para os Agregadores
        fog_strategy = FogFedAvg(
            min_available_clients=len(EDGE_SERVERS),
            min_fit_clients=fog_config["min_edges"],
            fraction_fit=1.0,
            min_evaluate_clients=0, # Não avaliamos durante o fit
            fraction_evaluate=0.0,
            evaluate_metrics_aggregation_fn=weighted_average,
            # ✅ CORREÇÃO: Inicia o treino dos Agregadores com o modelo do Cloud
            initial_parameters=fl.common.ndarrays_to_parameters(parameters)
        )
        
        # 3. Inicia um servidor de 1 rodada para os Agregadores
        fl.server.start_server(
            server_address=FOG_SERVER_ADDRESS,
            config=fl.server.ServerConfig(num_rounds=1),
            strategy=fog_strategy,
            # ✅ CORREÇÃO: Impede o gRPC de reiniciar
            server=fl.server.Server(client_manager=fl.server.SimpleClientManager(), strategy=fog_strategy)
        )
        
        # 4. 'fog_parameters' foi atualizado pela estratégia (FogFedAvg)
        print(f"🌫️  Fog [FIT]: Agregação local concluída. Enviando para o Cloud.")
        
        # 5. Converte e retorna os parâmetros recém-agregados
        if fog_parameters is not None:
            self.parameters = fl.common.parameters_to_ndarrays(fog_parameters)
            return self.parameters, 1, {"fog_layer": True}
        else:
            return self.parameters, 1, {"fog_layer": True}
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]):
        global fog_real_accuracy, fog_real_loss
        
        print(f"🌫️  Fog [EVAL]: Recebeu modelo do Cloud. Pedindo avaliação aos Agregadores...")
        # 1. Define a estratégia de avaliação para os Agregadores
        fog_eval_strategy = FogFedAvg(
            min_available_clients=len(EDGE_SERVERS),
            min_fit_clients=0, # Sem fit
            fraction_fit=0.0,
            min_evaluate_clients=fog_config["min_edges"],
            fraction_evaluate=1.0,
            evaluate_metrics_aggregation_fn=weighted_average,
            # ✅ CORREÇÃO: Passa o modelo do Cloud para avaliação
            initial_parameters=fl.common.ndarrays_to_parameters(parameters) 
        )
        
        # 2. Inicia um servidor de 1 rodada (APENAS AVALIAÇÃO)
        fl.server.start_server(
            server_address=FOG_SERVER_ADDRESS,
            config=fl.server.ServerConfig(num_rounds=1), # 1 rodada de avaliação
            strategy=fog_eval_strategy,
            server=fl.server.Server(client_manager=fl.server.SimpleClientManager(), strategy=fog_eval_strategy)
        )
        
        # 3. 'fog_real_accuracy' foi atualizada pela estratégia
        print(f"🌫️  Fog [EVAL]: Repassando Acurácia Real ({fog_real_accuracy:.4f}) para o Cloud")
        return fog_real_loss, 1, {"accuracy": fog_real_accuracy, "fog_layer": True}

# Conecta ao servidor (Cloud)
try:
    print(f"🌫️  Fog Server: Conectando ao Cloud em {args.cloud_ip}...")
    fl.client.start_client(
        server_address=args.cloud_ip,
        client=FogClient().to_client(),
    )
    
    print(f"🌫️  Fog Server finalizado")
    
except Exception as e:
    print(f"❌ Erro no Fog Server: {e}")
    import traceback
    traceback.print_exc()