import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader  # ✅ FALTANDO ESTE IMPORT!
from typing import Dict, List, Optional, Tuple
import argparse
import numpy as np
from collections import OrderedDict
import time
import json
import os

# ✅ IMPORTS CORRETOS
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_anomaly import AnomalyDetectionCNN, SimpleAnomalyDetector
from core.dataset_anomaly import load_anomaly_data

# ✅ CORREÇÃO DAS MÉTRICAS (com fallback)
try:
    from metrics.advanced_metrics import AdvancedMetrics
    import pandas as pd
except ImportError:
    print("⚠️  Métricas avançadas não encontradas, usando versão simples...")
    
    class AdvancedMetrics:
        def calculate_comprehensive_metrics(self, y_true, y_pred, y_scores):
            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
            try:
                accuracy = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, average='macro')
                precision = precision_score(y_true, y_pred, average='macro')
                recall = recall_score(y_true, y_pred, average='macro')
                try:
                    auc = roc_auc_score(y_true, y_scores)
                except:
                    auc = 0.5
                return {
                    'accuracy': accuracy, 'f1_score': f1, 'precision': precision,
                    'recall': recall, 'auc': auc
                }
            except:
                return {'accuracy': 0.5, 'f1_score': 0.5, 'precision': 0.5, 'recall': 0.5, 'auc': 0.5}
        
        def convergence_analysis(self, accuracy_history):
            if not accuracy_history:
                return {'converged': False, 'rounds_to_converge': None}
            return {'converged': True, 'rounds_to_converge': len(accuracy_history)}

# Configurações
parser = argparse.ArgumentParser(description="Servidor Global para Detecção de Anomalias")
parser.add_argument("--dataset", type=str, default="CIFAR10", choices=["MVTec", "CIFAR10", "Custom"])
parser.add_argument("--rounds", type=int, default=10)
parser.add_argument("--aggregators", type=int, default=2)
parser.add_argument("--use-transfer-learning", type=bool, default=True)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(args.seed)

# MÉTRICAS GLOBAIS EXPANDIDAS
global_metrics = {
    "round_times": [],
    "communication_costs": [],
    "accuracies": [],
    "losses": [],
    "f1_scores": [],
    "auc_scores": [],
    "convergence_history": []
}

def get_evaluate_fn():
    """Função de avaliação centralizada expandida"""
    _, test_dataset, model_class, model_config = load_anomaly_data(
        args.dataset, num_clients=1, normal_class=0,
        use_transfer_learning=args.use_transfer_learning
    )
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Inicializa métricas avançadas
    advanced_metrics = AdvancedMetrics()
    
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict):
        start_time = time.time()
        
        model = model_class(**model_config).to(DEVICE)
        
        # Carrega parâmetros
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        
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
                images = images.to(DEVICE)
                is_anomaly = is_anomaly.to(DEVICE)
                
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
        
        # MÉTRICAS AVANÇADAS
        comprehensive_metrics = advanced_metrics.calculate_comprehensive_metrics(
            all_labels, all_predictions, all_scores
        )
        
        accuracy = comprehensive_metrics['accuracy']
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        
        # Calcula custo de comunicação
        comm_cost = sum(p.nbytes for p in parameters)
        round_time = time.time() - start_time
        
        # ANÁLISE DE CONVERGÊNCIA
        global_metrics["accuracies"].append(accuracy)
        convergence_info = advanced_metrics.convergence_analysis(global_metrics["accuracies"])
        
        # Atualiza métricas globais
        global_metrics["round_times"].append(round_time)
        global_metrics["communication_costs"].append(comm_cost)
        global_metrics["losses"].append(avg_loss)
        global_metrics["f1_scores"].append(comprehensive_metrics['f1_score'])
        global_metrics["auc_scores"].append(comprehensive_metrics['auc'])
        
        print("=" * 70)
        print(f"🌐 SERVIDOR GLOBAL - Rodada {server_round}")
        print(f"📊 Loss: {avg_loss:.4f} | Acurácia: {accuracy:.4f}")
        print(f"🎯 F1-Score: {comprehensive_metrics['f1_score']:.4f} | AUC: {comprehensive_metrics['auc']:.4f}")
        print(f"⏱️  Tempo: {round_time:.2f}s | Comm: {comm_cost/1024/1024:.2f} MB")
        print(f"📈 Convergência: Rodada {convergence_info['rounds_to_converge'] if convergence_info['converged'] else 'N/A'}")
        print(f"📋 Precisão: {comprehensive_metrics['precision']:.4f} | Recall: {comprehensive_metrics['recall']:.4f}")
        print("=" * 70)
        
        # Salva métricas a cada rodada
        if server_round % 3 == 0:
            os.makedirs("metrics", exist_ok=True)
            with open(f"metrics/global_metrics_seed_{args.seed}.json", "w") as f:
                json.dump(global_metrics, f, indent=2)
        
        return avg_loss, {
            "accuracy": accuracy,
            "f1_score": comprehensive_metrics['f1_score'], 
            "auc": comprehensive_metrics['auc'],
            "precision": comprehensive_metrics['precision'],
            "recall": comprehensive_metrics['recall'],
            "loss": avg_loss,
            "communication_cost": comm_cost,
            "round_time": round_time,
            "convergence_round": convergence_info['rounds_to_converge']
        }
    
    return evaluate

# Estratégia com FedAvg
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    min_fit_clients=args.aggregators,
    fraction_evaluate=1.0,
    min_evaluate_clients=args.aggregators,
    min_available_clients=args.aggregators,
    evaluate_fn=get_evaluate_fn(),
)

print("🚀 INICIANDO SERVIDOR GLOBAL PARA DETECÇÃO DE ANOMALIAS")
print(f"📊 Dataset: {args.dataset}")
print(f"🔄 Rodadas: {args.rounds}")
print(f"🏢 Agregadores: {args.aggregators}")
print(f"🎯 Transfer Learning: {args.use_transfer_learning}")
print(f"🔢 Seed: {args.seed}")

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=args.rounds),
    strategy=strategy,
)

# Salva métricas finais
os.makedirs("metrics", exist_ok=True)
with open(f"metrics/final_global_metrics_seed_{args.seed}.json", "w") as f:
    json.dump(global_metrics, f, indent=2)
print(f"💾 Métricas salvas em: metrics/final_global_metrics_seed_{args.seed}.json")