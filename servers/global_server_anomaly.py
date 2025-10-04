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

# ✅ NOVOS ARGUMENTOS PARA BASELINES
parser = argparse.ArgumentParser(description="Servidor Global com Baselines")
parser.add_argument("--rounds", type=int, default=20)
parser.add_argument("--min-clients", type=int, default=10)
parser.add_argument("--architecture", type=str, default="hierarchical", 
                   choices=["hierarchical", "flat", "centralized"])
parser.add_argument("--scenario", type=str, default="small", 
                   choices=["small", "medium", "large"])
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()

# ✅ CONFIGURAÇÕES POR CENÁRIO
SCENARIO_CONFIGS = {
    "small": {"total_clients": 50, "min_clients": 10},
    "medium": {"total_clients": 100, "min_clients": 20}, 
    "large": {"total_clients": 200, "min_clients": 40}
}

scenario_config = SCENARIO_CONFIGS[args.scenario]
args.min_clients = scenario_config["min_clients"]

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

# ✅ ESTRATÉGIAS PARA DIFERENTES BASELINES
def get_strategy(architecture, tracker):
    if architecture == "flat":
        # Flat-FedAvg (sem hierarquia)
        return fl.server.strategy.FedAvg(
            fraction_fit=0.3,
            min_fit_clients=args.min_clients,
            fraction_evaluate=0.3,
            min_evaluate_clients=args.min_clients,
            min_available_clients=scenario_config["total_clients"],
        )
    elif architecture == "centralized":
        # Simulação de centralizado - todos os clientes participam
        return fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            min_fit_clients=scenario_config["total_clients"],
            fraction_evaluate=1.0,
            min_evaluate_clients=scenario_config["total_clients"],
            min_available_clients=scenario_config["total_clients"],
        )
    else:
        # Hierárquico padrão
        return fl.server.strategy.FedAvg(
            fraction_fit=0.4,
            min_fit_clients=args.min_clients,
            fraction_evaluate=0.4,
            min_evaluate_clients=args.min_clients,
            min_available_clients=scenario_config["total_clients"],
        )

tracker = ExperimentTracker()
start_time = time.time()

print("🚀 SERVIDOR GLOBAL INICIADO")
print(f"📊 Cenário: {args.scenario} ({scenario_config['total_clients']} clientes)")
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
    
    # Processa resultados finais
    tracker.metrics["total_training_time"] = time.time() - start_time
    
    # Salva resultados
    os.makedirs("results", exist_ok=True)
    results_file = f"results/{args.architecture}_{args.scenario}_seed_{args.seed}.json"
    
    results = {
        "architecture": args.architecture,
        "scenario": args.scenario,
        "total_clients": scenario_config["total_clients"],
        "metrics": tracker.metrics,
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
