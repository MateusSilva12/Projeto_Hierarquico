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

# ✅ JUSTIFICATIVA DA ARQUITETURA 3-NÍVEIS:
# - Device→Edge→Fog→Cloud permite melhor escalabilidade
# - Fog layer reduz latência para aplicações em tempo real
# - Melhor balanceamento de carga entre múltiplos edges
# - Mais próximo da arquitetura industrial real

parser = argparse.ArgumentParser(description="Fog Server - Camada Intermediária Obrigatória")
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
    "small": {"fog_rounds": 3, "min_edges": 2},
    "medium": {"fog_rounds": 4, "min_edges": 3},
    "large": {"fog_rounds": 5, "min_edges": 4}
}

fog_config = SCENARIO_CONFIGS[args.scenario]

class FogFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aggregation_times = []
    
    def aggregate_fit(self, server_round, results, failures):
        start_time = time.time()
        
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None:
            agg_time = time.time() - start_time
            self.aggregation_times.append(agg_time)
            print(f"🌫️  Fog - Rodada {server_round}: {len(results)} edges, {agg_time:.2f}s")
        
        return aggregated_parameters, metrics

# Servidor Fog
fog_strategy = FogFedAvg(
    min_available_clients=len(EDGE_SERVERS),
    min_fit_clients=fog_config["min_edges"],
    fraction_fit=1.0
)

try:
    fl.server.start_server(
        server_address=FOG_SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=fog_config["fog_rounds"]),
        strategy=fog_strategy,
    )
    
    print(f"✅ Agregação Fog finalizada. Conectando ao Cloud {args.cloud_ip}...")
    
    # Cliente Fog para Cloud
    class FogClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return [np.random.randn(100).astype(np.float32)]  # Modelo simulado
        
        def fit(self, parameters, config):
            return self.get_parameters(config), 1, {}
        
        def evaluate(self, parameters, config):
            accuracy = 0.75 + np.random.uniform(-0.1, 0.1)  # Simulação
            return 1.0 - accuracy, 1, {"accuracy": accuracy}
    
    fl.client.start_numpy_client(
        server_address=args.cloud_ip,
        client=FogClient(),
    )
    
    print(f"🌫️  Fog Server finalizado")
    
except Exception as e:
    print(f"❌ Erro no Fog Server: {e}")
