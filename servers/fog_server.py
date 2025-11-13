import flwr as fl
import torch
import argparse
import time
import json
import os
import sys
# ✅ CORREÇÃO: Imports adicionais
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

# Processa edges
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

# ✅ CORREÇÃO: Variáveis globais para armazenar parâmetros E acurácia real
fog_parameters = None
fog_real_accuracy = 0.0
fog_real_loss = 0.0

# ✅ CORREÇÃO: Função para agregar métricas reais dos Agregadores
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
            fog_parameters = aggregated_parameters
            agg_time = time.time() - start_time
            self.aggregation_times.append(agg_time)
            print(f"🌫️  Fog - Rodada {server_round}: {len(results)} edges, {agg_time:.2f}s")
        
        return aggregated_parameters, metrics

    # ✅ CORREÇÃO: Capturar a acurácia real agregada
    def aggregate_evaluate(self, server_round, results, failures):
        """Agrega métricas reais e as salva globalmente."""
        global fog_real_accuracy, fog_real_loss
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        
        if metrics and "accuracy" in metrics:
            # Salva a acurácia real para o FogClient usar
            fog_real_accuracy = metrics["accuracy"]
            fog_real_loss = loss if loss is not None else 0.0
        
        return loss, metrics

# ✅ CORREÇÃO: Servidor Fog usa a função de agregação
fog_strategy = FogFedAvg(
    min_available_clients=len(EDGE_SERVERS),
    min_fit_clients=fog_config["min_edges"],
    fraction_fit=1.0,
    min_evaluate_clients=fog_config["min_edges"],
    fraction_evaluate=1.0,
    evaluate_metrics_aggregation_fn=weighted_average # <-- ESSENCIAL
)

try:
    print("🚀 Iniciando servidor Fog...")
    fl.server.start_server(
        server_address=FOG_SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=fog_config["fog_rounds"]),
        strategy=fog_strategy,
    )
    
    print(f"✅ Agregação Fog finalizada. Conectando ao Cloud {args.cloud_ip}...")
    
    class FogClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            print(f"🌫️  Fog: Enviando parâmetros para Cloud")
            if fog_parameters is not None:
                try:
                    parameters_ndarrays = fl.common.parameters_to_ndarrays(fog_parameters)
                    print(f"✅ Fog: {len(parameters_ndarrays)} parâmetros convertidos")
                    return parameters_ndarrays
                except Exception as e:
                    print(f"⚠️  Erro ao converter fog_parameters: {e}")
                    model = SimpleAnomalyDetector(in_channels=3, num_classes=2)
                    return get_parameters(model)
            else:
                model = SimpleAnomalyDetector(in_channels=3, num_classes=2)
                return get_parameters(model)
        
        def fit(self, parameters, config):
            return self.get_parameters(config), 1, {"fog_layer": True}
        
        def evaluate(self, parameters, config):
            # ✅ CORREÇÃO: Parar de simular. Enviar a acurácia real.
            print(f"🌫️  Fog: Repassando Acurácia Real ({fog_real_accuracy:.4f}) para o Cloud")
            # Retorna a acurácia real que o aggregate_evaluate salvou
            return fog_real_loss, 1, {"accuracy": fog_real_accuracy, "fog_layer": True}
    
    fl.client.start_client(
        server_address=args.cloud_ip,
        client=FogClient().to_client(),
    )
    
    print(f"🌫️  Fog Server finalizado")
    
except Exception as e:
    print(f"❌ Erro no Fog Server: {e}")
    import traceback
    traceback.print_exc()