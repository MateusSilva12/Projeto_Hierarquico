import subprocess
import time
import yaml
import json
import os
from typing import Dict, List
import argparse

class ExperimentOrchestrator:
    def __init__(self, config_file: str = "configs/scenarios.yaml"):
        self.load_configs(config_file)
        self.results = {}
        
    def load_configs(self, config_file: str):
        """Carrega configurações dos experimentos"""
        with open(config_file, 'r') as f:
            self.configs = yaml.safe_load(f)
        
        self.scenarios = self.configs['scenarios']
        self.hardware_profiles = self.configs['hardware_profiles']
        self.seeds = self.configs['experiment_settings']['seeds']
        
    def run_experiments(self):
        """Executa todos os experimentos definidos"""
        print("🚀 INICIANDO EXPERIMENTOS FEDERADOS")
        print("=" * 60)
        
        for scenario_name, scenario_config in self.scenarios.items():
            print(f"\n🎯 EXECUTANDO CENÁRIO: {scenario_name}")
            print(f"📊 Clientes: {scenario_config['total_clients']}, "
                  f"Edges: {scenario_config['edge_servers']}, "
                  f"Rodadas: {scenario_config['global_rounds']}")
            
            scenario_results = []
            
            for seed in range(self.seeds):
                print(f"\n🌱 Seed {seed + 1}/{self.seeds}")
                result = self.run_single_experiment(scenario_name, scenario_config, seed)
                scenario_results.append(result)
                time.sleep(10)  # Intervalo entre seeds
            
            self.results[scenario_name] = scenario_results
            self.save_scenario_results(scenario_name, scenario_results)
        
        self.generate_final_report()
    
    def run_single_experiment(self, scenario_name: str, scenario_config: dict, seed: int):
        """Executa um único experimento"""
        start_time = time.time()
        
        # Inicia servidor global
        global_server_proc = self.start_global_server(scenario_config, seed)
        time.sleep(5)
        
        # Inicia servidores edge
        edge_procs = self.start_edge_servers(scenario_config, seed)
        time.sleep(10)
        
        # Inicia clientes
        client_procs = self.start_clients(scenario_name, scenario_config, seed)
        
        # Aguarda conclusão
        self.wait_for_completion(global_server_proc, scenario_config['global_rounds'])
        
        # Coleta resultados
        experiment_time = time.time() - start_time
        metrics = self.collect_metrics(scenario_name, seed)
        
        # Encerra processos
        self.cleanup_processes([global_server_proc] + edge_procs + client_procs)
        
        return {
            'scenario': scenario_name,
            'seed': seed,
            'duration': experiment_time,
            'metrics': metrics
        }
    
    def start_global_server(self, scenario_config: dict, seed: int):
        """Inicia servidor global"""
        cmd = [
            'python', 'servers/global_server_anomaly.py',
            '--rounds', str(scenario_config['global_rounds']),
            '--aggregators', str(scenario_config['edge_servers']),
            '--seed', str(seed)
        ]
        return subprocess.Popen(cmd)
    
    def start_edge_servers(self, scenario_config: dict, seed: int):
        """Inicia servidores edge"""
        processes = []
        base_port = 8040
        
        for i in range(scenario_config['edge_servers']):
            cmd = [
                'python', 'servers/aggregator_anomaly.py',
                '--port', str(base_port + i),
                '--num-local-clients', str(scenario_config['clients_per_edge']),
                '--num-local-rounds', str(scenario_config['local_rounds']),
                '--seed', str(seed + i)
            ]
            proc = subprocess.Popen(cmd)
            processes.append(proc)
            time.sleep(2)  # Intervalo entre edges
        
        return processes
    
    def start_clients(self, scenario_name: str, scenario_config: dict, seed: int):
        """Inicia clientes com perfis de heterogeneidade"""
        processes = []
        base_port = 8040
        clients_per_edge = scenario_config['clients_per_edge']
        total_clients = scenario_config['total_clients']
        
        # Distribuição de perfis (20% high, 50% medium, 20% low, 10% straggler)
        profile_distribution = ['high_end'] * int(0.2 * total_clients) + \
                              ['medium'] * int(0.5 * total_clients) + \
                              ['low_end'] * int(0.2 * total_clients) + \
                              ['straggler'] * int(0.1 * total_clients)
        
        client_id = 0
        for edge_idx in range(scenario_config['edge_servers']):
            server_ip = f"127.0.0.1:{base_port + edge_idx}"
            
            for _ in range(clients_per_edge):
                if client_id >= total_clients:
                    break
                    
                profile = profile_distribution[client_id % len(profile_distribution)]
                
                cmd = [
                    'python', 'core/cliente_anomaly_sgd.py',
                    '--server-ip', server_ip,
                    '--partition-id', str(client_id),
                    '--total-clients', str(total_clients),
                    '--scenario', scenario_name,
                    '--client-profile', profile,
                    '--seed', str(seed + client_id)
                ]
                
                # Adiciona emulação de rede baseada no perfil
                if profile == 'high_end':
                    cmd.extend(['--network-latency', '10', '--packet-loss', '0.1'])
                elif profile == 'medium':
                    cmd.extend(['--network-latency', '50', '--packet-loss', '0.5'])
                elif profile == 'low_end':
                    cmd.extend(['--network-latency', '100', '--packet-loss', '1.0'])
                else:  # straggler
                    cmd.extend(['--network-latency', '200', '--packet-loss', '5.0'])
                
                proc = subprocess.Popen(cmd)
                processes.append(proc)
                client_id += 1
                
                time.sleep(0.5)  # Intervalo entre clientes
        
        return processes
    
    def wait_for_completion(self, global_server_proc, expected_rounds: int):
        """Aguarda conclusão do experimento"""
        max_wait_time = expected_rounds * 120  # 2 minutos por rodada
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            if global_server_proc.poll() is not None:
                break
            time.sleep(30)  # Verifica a cada 30 segundos
        
        # Timeout - encerra processo
        if global_server_proc.poll() is None:
            global_server_proc.terminate()
            print("⏰ Timeout - Experimento interrompido")
    
    def collect_metrics(self, scenario_name: str, seed: int):
        """Coleta métricas do experimento"""
        metrics_files = [
            f"metrics/global_metrics_seed_{seed}.json",
            f"metrics/aggregator_*_seed_{seed}.json", 
            f"metrics/client_*_scenario_{scenario_name}_seed_{seed}.json"
        ]
        
        collected_metrics = {}
        # Lógica para agregar métricas de todos os arquivos
        # Implementação detalhada no advanced_metrics.py
        
        return collected_metrics
    
    def cleanup_processes(self, processes: list):
        """Encerra todos os processos"""
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
                proc.wait()
    
    def save_scenario_results(self, scenario_name: str, results: list):
        """Salva resultados do cenário"""
        os.makedirs("results", exist_ok=True)
        filename = f"results/{scenario_name}_results.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Resultados salvos em: {filename}")
    
    def generate_final_report(self):
        """Gera relatório final comparativo"""
        from metrics.statistical_analysis import StatisticalAnalyzer
        
        analyzer = StatisticalAnalyzer()
        report = analyzer.generate_comprehensive_report(self.results)
        
        with open("results/final_experiment_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        print("📊 RELATÓRIO FINAL GERADO: results/final_experiment_report.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orquestrador de Experimentos Federados")
    parser.add_argument("--config", type=str, default="configs/scenarios.yaml", 
                       help="Arquivo de configuração")
    args = parser.parse_args()
    
    orchestrator = ExperimentOrchestrator(args.config)
    orchestrator.run_experiments()