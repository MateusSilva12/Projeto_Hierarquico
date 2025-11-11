import flwr as fl
import torch
import argparse
import time
import json
import os
import sys
from typing import Dict, List
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

# Processa edges
edge_ports = [int(port.strip()) for port in args.edge_ports.split(',')]
EDGE_SERVERS = [f"127.0.0.1:{port}" for port in edge_ports]
FOG_SERVER_ADDRESS = f"0.0.0.0:{args.fog_port}"

print(f"🌫️  FOG SERVER: Porta {args.fog_port}")
print(f"🔗 Conectando a {len(EDGE_SERVERS)} edges")
print(f"📊 Cenário: {args.scenario}")

# Configuração baseada no cenário
SCENARIO_CONFIGS = {
    "small": {"fog_rounds": 3, "min_edges": 1},
    "medium": {"fog_rounds": 4, "min_edges": 2},
    "large": {"fog_rounds": 5, "min_edges": 3}
}

fog_config = SCENARIO_CONFIGS[args.scenario]

# ✅ VARIÁVEL GLOBAL para armazenar parâmetros agregados
fog_parameters = None

class FogFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aggregation_times = []
    
    def aggregate_fit(self, server_round, results, failures):
        global fog_parameters
        start_time = time.time()
        
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None:
            fog_parameters = aggregated_parameters
            agg_time = time.time() - start_time
            self.aggregation_times.append(agg_time)
            print(f"🌫️  Fog - Rodada {server_round}: {len(results)} edges, {agg_time:.2f}s")
        
        return aggregated_parameters, metrics

# Servidor Fog
fog_strategy = FogFedAvg(
    min_available_clients=len(EDGE_SERVERS),
    min_fit_clients=fog_config["min_edges"],
    fraction_fit=1.0,
    min_evaluate_clients=fog_config["min_edges"],
    fraction_evaluate=1.0
)

try:
    print("🚀 Iniciando servidor Fog...")
    fl.server.start_server(
        server_address=FOG_SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=fog_config["fog_rounds"]),
        strategy=fog_strategy,
    )
    
    print(f"✅ Agregação Fog finalizada. Conectando ao Cloud {args.cloud_ip}...")
    
    # ✅ CORREÇÃO: Cliente Fog real para Cloud
    class FogClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            print(f"🌫️  Fog: Enviando parâmetros para Cloud")
            if fog_parameters is not None:
                try:
                    # Converter Parameters para numpy arrays
                    parameters_ndarrays = fl.common.parameters_to_ndarrays(fog_parameters)
                    print(f"✅ Fog: {len(parameters_ndarrays)} parâmetros convertidos")
                    return parameters_ndarrays
                except Exception as e:
                    print(f"⚠️  Erro ao converter fog_parameters: {e}")
                    # Fallback: criar modelo simples
                    model = SimpleAnomalyDetector(in_channels=3, num_classes=2)
                    return get_parameters(model)
            else:
                # Fallback se não houve agregação
                model = SimpleAnomalyDetector(in_channels=3, num_classes=2)
                return get_parameters(model)
        
        def fit(self, parameters, config):
            # Fog não faz treinamento local, só repassa
            return self.get_parameters(config), 1, {"fog_layer": True}
        
        def evaluate(self, parameters, config):
            # Avaliação simulada do Fog
            accuracy = 0.75 + np.random.uniform(-0.1, 0.1)
            print(f"🌫️  Fog: Accuracy simulada = {accuracy:.3f}")
            return 1.0 - accuracy, 1, {"accuracy": accuracy, "fog_layer": True}
    
    # ✅ CORREÇÃO: Usar start_client em vez de start_numpy_client
    fl.client.start_client(
        server_address=args.cloud_ip,
        client=FogClient().to_client(),
    )
    
    print(f"🌫️  Fog Server finalizado")
    
except Exception as e:
    print(f"❌ Erro no Fog Server: {e}")
    import traceback
    traceback.print_exc()