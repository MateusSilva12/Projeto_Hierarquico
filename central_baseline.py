# Arquivo: server.py (VERSÃO FINAL E ROBUSTA)

import flwr as fl
import torch
from torch.utils.data import DataLoader
from typing import Dict
import matplotlib.pyplot as plt
import argparse
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from model import UniversalNet, PretrainedResNet
from dataset import load_data
from utils import set_parameters

# Configuração do Servidor
parser = argparse.ArgumentParser(description="Servidor Flower com Acurácia e AUC")
parser.add_argument("--dataset", type=str, default="CIFAR10", choices=["MNIST", "CIFAR10", "CIFAR100"])
parser.add_argument("--rounds", type=int, default=20)
parser.add_argument("--clients", type=int, default=5)
args = parser.parse_args()
USE_TRANSFER_LEARNING = True
USE_DATA_AUGMENTATION = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Função de Avaliação com Acurácia e AUC
def get_evaluate_fn():
    _, test_dataset, ModelClass, model_init_params = load_data(
        args.dataset, args.clients, USE_DATA_AUGMENTATION, USE_TRANSFER_LEARNING
    )
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict):
        model = ModelClass(**model_init_params).to(DEVICE)
        set_parameters(model, parameters)
        criterion = torch.nn.CrossEntropyLoss()
        model.eval()
        
        total_loss, correct, total = 0.0, 0, 0
        all_labels, all_probs = [], []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                probs = F.softmax(outputs, dim=1)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = correct / total
        roc_auc = 0.0
        try:
            roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        except ValueError as e:
            print(f"Não foi possível calcular AUC: {e}")

        print("-" * 40)
        print(f"AVALIAÇÃO NO SERVIDOR ({args.dataset}) - Rodada {server_round}")
        print(f"Acurácia: {accuracy:.4f} | AUC: {roc_auc:.4f}")
        print("-" * 40)
        
        return total_loss / len(test_loader), {"accuracy": accuracy, "roc_auc": roc_auc}
    return evaluate

# ✅ CORRIGIDO: Função de plotagem completa
def plotar_resultados(history, nome_dataset, sufixo):
    acc_data = history.metrics_centralized.get("accuracy")
    auc_data = history.metrics_centralized.get("roc_auc")
    
    if not acc_data:
        print("⚠️  Nenhum dado de acurácia disponível para plotagem")
        return
    
    rodadas = [item[0] for item in acc_data]
    valores_acc = [item[1] for item in acc_data]
    
    fig, ax1 = plt.subplots(figsize=(14, 8))
    color = 'tab:blue'
    ax1.set_xlabel('Rodada')
    ax1.set_ylabel('Acurácia', color=color)
    ax1.plot(rodadas, valores_acc, marker='o', color=color, label=f"Acurácia ({nome_dataset})")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--')
    
    if auc_data:
        valores_auc = [item[1] for item in auc_data]
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('AUC (OvR)', color=color)
        ax2.plot(rodadas, valores_auc, marker='s', linestyle='--', color=color, label=f"AUC ({nome_dataset})")
        ax2.tick_params(axis='y', labelcolor=color)
    
    fig.suptitle(f"Métricas Globais ({sufixo})", fontsize=16)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels() if auc_data else ([], [])
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    plt.xticks(rodadas)
    plt.savefig(f"grafico_metricas_{nome_dataset}_{sufixo}.png")
    print(f"\nGráfico salvo como 'grafico_metricas_{nome_dataset}_{sufixo}.png'")

# Inicialização do Servidor
sufixo_nome = "Final_Run"
print(f"\nINICIANDO SERVIDOR PARA O EXPERIMENTO: {sufixo_nome}")

strategy = fl.server.strategy.FedProx(
    fraction_fit=1.0,
    min_fit_clients=args.clients,
    fraction_evaluate=0.0,
    min_available_clients=args.clients,
    evaluate_fn=get_evaluate_fn(),
    proximal_mu=0.01 
)

history = fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=args.rounds),
    strategy=strategy,
)

print("\n--- TREINAMENTO DISTRIBUÍDO FINALIZADO ---")
plotar_resultados(history, args.dataset, sufixo_nome)