import json
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def load_all_metrics():
    """Carrega todas as métricas dos experimentos"""
    metrics_files = glob("*metrics_seed_*.json")
    
    results = {}
    for file in metrics_files:
        with open(file, 'r') as f:
            data = json.load(f)
        
        # Extrai informações do nome do arquivo
        if "global" in file:
            key = "global_" + file.split("_seed_")[1].replace(".json", "")
            results[key] = data
        elif "aggregator" in file:
            key = "aggregator_" + file.split("_seed_")[1].replace(".json", "")
            results[key] = data
        elif "client" in file:
            key = "client_" + file.split("_seed_")[1].replace(".json", "")
            results[key] = data
    
    return results

def analyze_convergence(global_metrics):
    """Analisa convergência dos modelos"""
    convergence_data = {}
    
    for exp_name, metrics in global_metrics.items():
        if "global" not in exp_name:
            continue
            
        accuracies = metrics.get('accuracies', [])
        losses = metrics.get('losses', [])
        
        if len(accuracies) > 0:
            convergence_data[exp_name] = {
                'final_accuracy': accuracies[-1],
                'best_accuracy': max(accuracies),
                'convergence_round': np.argmax(accuracies) + 1,
                'final_loss': losses[-1] if losses else None,
                'avg_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies)
            }
    
    return pd.DataFrame.from_dict(convergence_data, orient='index')

def analyze_communication(metrics):
    """Analisa custos de comunicação"""
    comm_data = []
    
    for exp_name, data in metrics.items():
        if "global" in exp_name:
            comm_costs = data.get('communication_costs', [])
            if comm_costs:
                total_comm = sum(comm_costs) / (1024 * 1024)  # Convert to MB
                comm_data.append({
                    'experiment': exp_name,
                    'total_communication_mb': total_comm,
                    'avg_round_comm_mb': np.mean(comm_costs) / (1024 * 1024),
                    'total_rounds': len(comm_costs)
                })
    
    return pd.DataFrame(comm_data)

def analyze_client_behavior(client_metrics):
    """Analisa comportamento dos clientes"""
    client_data = []
    
    for exp_name, data in client_metrics.items():
        if "client" in exp_name:
            client_data.append({
                'client_id': exp_name,
                'participation_rounds': data.get('participation_rounds', 0),
                'avg_training_time': np.mean(data.get('training_times', [0])),
                'avg_accuracy': np.mean(data.get('accuracies', [0])),
                'total_comm_mb': sum(data.get('communication_sizes', [0])) / (1024 * 1024)
            })
    
    return pd.DataFrame(client_data)

def statistical_analysis(convergence_df):
    """Realiza análise estatística"""
    stats_results = {}
    
    # Agrupa por escala
    scales = {}
    for exp_name in convergence_df.index:
        scale = int(exp_name.split('_')[0])
        if scale not in scales:
            scales[scale] = []
        scales[scale].append(convergence_df.loc[exp_name, 'final_accuracy'])
    
    # Teste t entre escalas
    scale_pairs = []
    for i, scale1 in enumerate(scales.keys()):
        for scale2 in list(scales.keys())[i+1:]:
            t_stat, p_value = stats.ttest_ind(scales[scale1], scales[scale2])
            scale_pairs.append({
                'comparison': f"{scale1} vs {scale2}",
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            })
    
    return pd.DataFrame(scale_pairs)

def generate_report():
    """Gera relatório completo dos experimentos"""
    print("📊 GERANDO RELATÓRIO DE EXPERIMENTOS")
    print("=" * 60)
    
    all_metrics = load_all_metrics()
    
    # Análise de convergência
    convergence_df = analyze_convergence(all_metrics)
    print("\n🎯 ANÁLISE DE CONVERGÊNCIA:")
    print(convergence_df)
    
    # Análise de comunicação
    comm_df = analyze_communication(all_metrics)
    print("\n📡 ANÁLISE DE COMUNICAÇÃO:")
    print(comm_df)
    
    # Análise estatística
    stats_df = statistical_analysis(convergence_df)
    print("\n📈 ANÁLISE ESTATÍSTICA:")
    print(stats_df)
    
    # Salva resultados
    with open("experiment_analysis_report.json", "w") as f:
        report = {
            "convergence_analysis": convergence_df.to_dict(),
            "communication_analysis": comm_df.to_dict(),
            "statistical_analysis": stats_df.to_dict()
        }
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Relatório salvo em: experiment_analysis_report.json")
    
    # Gera gráficos
    generate_plots(convergence_df, comm_df)

def generate_plots(convergence_df, comm_df):
    """Gera gráficos dos resultados"""
    plt.figure(figsize=(15, 10))
    
    # Gráfico 1: Acurácia por escala
    plt.subplot(2, 2, 1)
    convergence_df['scale'] = [int(name.split('_')[0]) for name in convergence_df.index]
    sns.boxplot(data=convergence_df, x='scale', y='final_accuracy')
    plt.title('Acurácia Final por Escala de Clientes')
    plt.xlabel('Número de Clientes')
    plt.ylabel('Acurácia')
    
    # Gráfico 2: Comunicação por escala
    plt.subplot(2, 2, 2)
    comm_df['scale'] = [int(name.split('_')[0]) for name in comm_df['experiment']]
    sns.barplot(data=comm_df, x='scale', y='total_communication_mb')
    plt.title('Custo Total de Comunicação por Escala')
    plt.xlabel('Número de Clientes')
    plt.ylabel('Comunicação Total (MB)')
    
    # Gráfico 3: Tempo de convergência
    plt.subplot(2, 2, 3)
    sns.scatterplot(data=convergence_df, x='scale', y='convergence_round', size='final_accuracy')
    plt.title('Rodadas até Convergência vs Escala')
    plt.xlabel('Número de Clientes')
    plt.ylabel('Rodadas até Convergência')
    
    # Gráfico 4: Distribuição de acurácias
    plt.subplot(2, 2, 4)
    convergence_df['final_accuracy'].hist(bins=20, alpha=0.7)
    plt.title('Distribuição de Acurácias Finais')
    plt.xlabel('Acurácia')
    plt.ylabel('Frequência')
    
    plt.tight_layout()
    plt.savefig('experiment_results_analysis.png', dpi=300, bbox_inches='tight')
    print("📊 Gráficos salvos em: experiment_results_analysis.png")

if __name__ == "__main__":
    generate_report()