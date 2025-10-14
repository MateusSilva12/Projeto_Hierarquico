import flwr as fl
import torch
import argparse
import time
import json
import os
import sys
from typing import Dict, List
import numpy as np
import psutil

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.model_anomaly import SimpleAnomalyDetector
from core.dataset_anomaly import load_anomaly_data

# ✅ CORREÇÃO: Função para calcular custo de comunicação
def calculate_communication_cost(parameters):
    """Calcula o custo de comunicação em bytes"""
    if parameters is None:
        return 0
    total_bytes = 0
    for param in parameters:
        total_bytes += param.nbytes
    return total_bytes

# ✅ NOVOS ARGUMENTOS PARA BASELINES
parser = argparse.ArgumentParser(description="Servidor Global com Baselines")
parser.add_argument("--rounds", type=int, default=20)
parser.add_argument("--min-clients", type=int, default=None, help="Número mínimo de clientes para uma rodada de treinamento. Se não especificado, usa o valor do cenário.")
parser.add_argument("--architecture", type=str, default="hierarchical", 
                   choices=["hierarchical", "flat", "centralized"])
parser.add_argument("--scenario", type=str, default="small", 
                   choices=["small", "medium", "large", "custom"])
parser.add_argument("--total-clients", type=int, default=None, help="Número total de clientes esperados. Se não especificado, usa o valor do cenário.")
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()

# ✅ CONFIGURAÇÕES POR CENÁRIO
SCENARIO_CONFIGS = {
    "small": {"total_clients": 50, "min_clients": 10},
    "medium": {"total_clients": 100, "min_clients": 20}, 
    "large": {"total_clients": 200, "min_clients": 40},
    "custom": {"total_clients": 4, "min_clients": 2}
}

scenario_config = SCENARIO_CONFIGS[args.scenario]

# Sobrescreve min_clients e total_clients se forem passados como argumentos
if args.min_clients is None:
    args.min_clients = scenario_config["min_clients"]
if args.total_clients is None:
    args.total_clients = scenario_config["total_clients"]

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
        
        # Detectar convergência (95% do máximo)
        if len(self.metrics["round_accuracies"]) > 5:
            max_acc = max(self.metrics["round_accuracies"])
            current_acc = self.metrics["round_accuracies"][-1]
            if current_acc >= 0.95 * max_acc and self.metrics["convergence_round"] is None:
                self.metrics["convergence_round"] = len(self.metrics["round_accuracies"])

# ✅ CORREÇÃO: Estratégias que usam o tracker
class TrackingStrategy(fl.server.strategy.FedAvg):
    def __init__(self, tracker, **kwargs):
        super().__init__(**kwargs)
        self.tracker = tracker
        self.round_start_time = None

    def aggregate_fit(self, server_round, results, failures):
        round_start_time = time.time()
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        
        # Calcula métricas da rodada
        round_time = time.time() - round_start_time
        comm_cost = calculate_communication_cost(aggregated_parameters)
        
        # Busca accuracy dos resultados (se disponível)
        accuracy = 0.0
        if results:
            # Tenta extrair accuracy do primeiro cliente
            first_client_metrics = results[0][1].metrics
            if first_client_metrics and "accuracy" in first_client_metrics:
                accuracy = first_client_metrics["accuracy"]
        
        # Registra no tracker
        self.tracker.record_round(accuracy, comm_cost, round_time)
        
        return aggregated_parameters, metrics

def get_strategy(architecture, tracker):
    base_config = {
        "fraction_evaluate": 0.3,
        "min_evaluate_clients": args.min_clients,
        "min_available_clients": args.total_clients,
    }
    
    if architecture == "flat":
        # Flat-FedAvg (sem hierarquia)
        return TrackingStrategy(
            tracker=tracker,
            fraction_fit=0.3,
            min_fit_clients=args.min_clients,
            **base_config
        )
    elif architecture == "centralized":
        # Simulação de centralizado - todos os clientes participam
        return TrackingStrategy(
            tracker=tracker,
            fraction_fit=1.0,
            min_fit_clients=args.total_clients,
            **base_config
        )
    else:
        # Hierárquico padrão
        return TrackingStrategy(
            tracker=tracker,
            fraction_fit=0.4,
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

try:
    strategy = get_strategy(args.architecture, tracker)
    
    # Configuração do servidor
    server_config = fl.server.ServerConfig(num_rounds=args.rounds)
    
    # Histórico para análise de convergência
    history = fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=server_config,
        strategy=strategy,
    )
    
    # ✅ CORREÇÃO: Preenche métricas finais do histórico Flower
    if history and hasattr(history, 'metrics_centralized'):
        for round_num, metrics in history.metrics_centralized.items():
            if 'accuracy' in metrics:
                accuracy = metrics['accuracy']
                if round_num - 1 < len(tracker.metrics["round_accuracies"]):
                    tracker.metrics["round_accuracies"][round_num - 1] = accuracy
    
    # Processa resultados finais
    tracker.metrics["total_training_time"] = time.time() - start_time
    
    # Salva resultados
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
