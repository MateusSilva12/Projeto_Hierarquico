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
import psutil

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.model_anomaly import SimpleAnomalyDetector
from core.dataset_anomaly import load_anomaly_data

parser = argparse.ArgumentParser(description="Servidor Global com Baselines")
parser.add_argument("--rounds", type=int, default=20)
parser.add_argument("--min-clients", type=int, default=None, help="Número mínimo de clientes para uma rodada de treinamento. Se não especificado, usa o valor do cenário.")
parser.add_argument("--architecture", type=str, default="hierarchical", 
                   choices=["hierarchical", "flat", "centralized"])
parser.add_argument("--scenario", type=str, default="small", 
                   choices=["small", "medium", "large", "custom"])
parser.add_argument("--total-clients", type=int, default=None, help="Número total de clientes esperados. Se não especificado, usa o valor do cenário.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--port", type=int, default=9080, help="Porta do servidor global")

args = parser.parse_args()

SCENARIO_CONFIGS = {
    "small": {"total_clients": 50, "min_clients": 10},
    "medium": {"total_clients": 100, "min_clients": 20}, 
    "large": {"total_clients": 200, "min_clients": 40},
    # ✅ CORREÇÃO: 'custom' espera os 2 Fogs
    "custom": {"total_clients": 2, "min_clients": 2} 
}

scenario_config = SCENARIO_CONFIGS[args.scenario]

if args.min_clients is None:
    args.min_clients = scenario_config["min_clients"]
if args.total_clients is None:
    args.total_clients = scenario_config["total_clients"]

# ✅ CORREÇÃO: Função para agregar métricas (como acurácia)
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Agrega métricas de avaliação (como acurácia) pela média ponderada."""
    # O FogClient envia '1' como num_examples, então isso é uma média simples.
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics if "accuracy" in m]
    examples = [num_examples for num_examples, m in metrics if "accuracy" in m]
    
    if not examples:
        return {}
        
    avg_accuracy = sum(accuracies) / sum(examples)
    return {"accuracy": avg_accuracy}

class ExperimentTracker:
    def __init__(self):
        self.metrics = {
            "round_accuracies": [],
            "communication_costs": [],
            "round_times": [],
            "convergence_round": None,
            "total_training_time": 0
        }
    
    def record_round(self, accuracy, comm_cost, round_time):
        self.metrics["round_accuracies"].append(accuracy)
        self.metrics["communication_costs"].append(comm_cost)
        self.metrics["round_times"].append(round_time)
        
        # ✅ CORREÇÃO: Evitar convergência falsa em 0.0
        if len(self.metrics["round_accuracies"]) > 5 and max(self.metrics["round_accuracies"]) > 0:
            max_acc = max(self.metrics["round_accuracies"])
            current_acc = self.metrics["round_accuracies"][-1]
            if current_acc >= 0.95 * max_acc and self.metrics["convergence_round"] is None:
                self.metrics["convergence_round"] = len(self.metrics["round_accuracies"])

def calculate_communication_cost(parameters):
    if parameters is None:
        return 0
    total_bytes = 0
    try:
        if hasattr(parameters, 'tensors'):
            parameters_list = fl.common.parameters_to_ndarrays(parameters)
        else:
            parameters_list = parameters
        for param in parameters_list:
            total_bytes += param.nbytes
    except Exception as e:
        print(f"⚠️  Erro ao calcular custo de comunicação: {e}")
        total_bytes = 0
    return total_bytes

# ✅ CORREÇÃO: Estratégia de Tracking modificada
class TrackingStrategy(fl.server.strategy.FedAvg):
    def __init__(self, tracker, **kwargs):
        super().__init__(**kwargs)
        self.tracker = tracker
        self.round_start_time = None

    def aggregate_fit(self, server_round, results, failures):
        # 1. Salva o tempo e custo de comunicação
        self.round_start_time = time.time()
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        comm_cost = calculate_communication_cost(aggregated_parameters)
        
        # Registra no tracker (a acurácia será 0.0 por enquanto, será atualizada no evaluate)
        self.tracker.record_round(accuracy=0.0, comm_cost=comm_cost, round_time=0.0) 
        
        return aggregated_parameters, metrics

    # ✅ CORREÇÃO: Nova função para capturar a acurácia da avaliação
    def aggregate_evaluate(self, server_round, results, failures):
        """Agrega métricas de avaliação e ATUALIZA o tracker."""
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        
        if aggregated_metrics and "accuracy" in aggregated_metrics:
            accuracy = aggregated_metrics["accuracy"]
            
            # ATUALIZA a acurácia da rodada que acabamos de registrar
            if server_round - 1 < len(self.tracker.metrics["round_accuracies"]):
                self.tracker.metrics["round_accuracies"][server_round - 1] = accuracy

            # Calcula o tempo total da rodada (Fit + Evaluate)
            round_time = time.time() - self.round_start_time if self.round_start_time else 0
            if server_round - 1 < len(self.tracker.metrics["round_times"]):
                self.tracker.metrics["round_times"][server_round - 1] = round_time
        
        return aggregated_loss, aggregated_metrics

def get_strategy(architecture, tracker):
    base_config = {
        "fraction_evaluate": 1.0, # ✅ CORREÇÃO: Avaliar em todos os Fogs (100%)
        "min_evaluate_clients": args.min_clients,
        "min_available_clients": args.total_clients,
        # ✅ CORREÇÃO: Passar a função de agregação de métricas
        "evaluate_metrics_aggregation_fn": weighted_average, 
    }
    
    if architecture == "flat":
        return TrackingStrategy(
            tracker=tracker,
            fraction_fit=0.3,
            min_fit_clients=args.min_clients,
            **base_config
        )
    elif architecture == "centralized":
        return TrackingStrategy(
            tracker=tracker,
            fraction_fit=1.0,
            min_fit_clients=args.total_clients,
            **base_config
        )
    else: # Hierárquico
        return TrackingStrategy(
            tracker=tracker,
            # ✅ CORREÇÃO: Usar 1.0 para treinar em ambos os Fogs (100%)
            fraction_fit=1.0, 
            min_fit_clients=args.min_clients,
            **base_config
        )

tracker = ExperimentTracker()
start_time = time.time()

print("🚀 SERVIDOR GLOBAL INICIADO")
print(f"📊 Cenário: {args.scenario} ({args.total_clients} clientes)")
print(f"🏗️  Arquitetura: {args.architecture}")
print(f"🔄 Rodadas: {args.rounds}")
print(f"👥 Mín. clientes: {args.min_clients}")
print(f"🔌 Porta: {args.port}")

try:
    strategy = get_strategy(args.architecture, tracker)
    
    server_config = fl.server.ServerConfig(num_rounds=args.rounds)
    server_address = f"0.0.0.0:{args.port}"
    
    history = fl.server.start_server(
        server_address=server_address,
        config=server_config,
        strategy=strategy,
    )
    
    tracker.metrics["total_training_time"] = time.time() - start_time
    
    os.makedirs("results", exist_ok=True)
    results_file = f"results/{args.architecture}_{args.scenario}_seed_{args.seed}.json"
    
    results = {
        "architecture": args.architecture,
        "scenario": args.scenario,
        "total_clients": args.total_clients,
        "metrics": tracker.metrics,
        "server_memory_usage_mb": psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024),
        "final_accuracy": tracker.metrics["round_accuracies"][-1] if tracker.metrics["round_accuracies"] else 0,
        "convergence_round": tracker.metrics["convergence_round"],
        "total_communication_mb": sum(tracker.metrics["communication_costs"]) / (1024 * 1024),
        "total_training_hours": tracker.metrics["total_training_time"] / 3600
    }
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Experimento concluído: {results_file}")
    print(f"📈 Acurácia final: {results['final_accuracy']:.4f}")
    print(f"🎯 Convergência: rodada {results['convergence_round']}")
    print(f"📊 Comunicação total: {results['total_communication_mb']:.2f} MB")
    
except Exception as e:
    print(f"❌ Erro no servidor: {e}")
    import traceback
    traceback.print_exc()