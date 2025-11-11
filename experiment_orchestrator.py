import subprocess
import time
import yaml
import json
import os
from typing import Dict, List
import argparse
import sys

class ExperimentOrchestrator:
    def __init__(self, config_file: str = "configs/scenarios.yaml"):
        self.configs = self.load_configs(config_file)
        
    def load_configs(self, config_file: str):
        """Carrega configurações"""
        try:
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        except:
            # Configuração padrão
            return {
                'scenarios': {
                    'small': {'total_clients': 20, 'edge_servers': 2, 'global_rounds': 10},
                    'medium': {'total_clients': 50, 'edge_servers': 3, 'global_rounds': 15}
                },
                'seeds': 3
            }
    
    def run_experiments(self):
        """Executa experimentos"""
        print("🚀 INICIANDO EXPERIMENTOS")
        
        for scenario_name, scenario_config in self.configs['scenarios'].items():
            print(f"\n🎯 CENÁRIO: {scenario_name}")
            print(f"👥 Clientes: {scenario_config['total_clients']}")
            
            for seed in range(self.configs['seeds']):
                print(f"🌱 Seed {seed + 1}")
                self.run_single_experiment(scenario_name, scenario_config, seed)
                time.sleep(10)
    
    def run_single_experiment(self, scenario_name: str, scenario_config: dict, seed: int):
        """Executa um experimento"""
        # Inicia servidor global
        server_proc = subprocess.Popen([
            sys.executable, "servers/global_server_anomaly.py",
            "--rounds", str(scenario_config['global_rounds']),
            "--min-clients", "2",
            "--seed", str(seed)
        ])
        
        time.sleep(5)
        
        # Inicia agregadores
        edge_procs = []
        base_port = 8081
        
        for i in range(min(2, scenario_config.get('edge_servers', 2))):
            proc = subprocess.Popen([
                sys.executable, "servers/aggregator_anomaly.py",
                "--port", str(base_port + i),
                "--server-ip", "127.0.0.1:8080",
                "--min-clients", "2",
                "--seed", str(seed + i)
            ])
            edge_procs.append(proc)
            time.sleep(2)
        
        time.sleep(5)
        
        # Inicia clientes
        client_procs = []
        total_clients = min(10, scenario_config['total_clients'])  # Limite para teste
        
        profiles = ['high_end', 'medium', 'low_end', 'straggler']
        
        for client_id in range(total_clients):
            profile = profiles[client_id % len(profiles)]
            server_port = 8081 + (client_id % 2)  # Distribui entre edges
            
            proc = subprocess.Popen([
                sys.executable, "clients/cliente_real.py",
                "--server-ip", f"127.0.0.1:{server_port}",
                "--client-id", str(client_id),
                "--profile", profile,
                "--seed", str(seed + client_id)
            ])
            client_procs.append(proc)
            time.sleep(0.5)
        
        # Aguarda
        max_wait = scenario_config['global_rounds'] * 60
        time.sleep(min(300, max_wait))
        
        # Encerra processos
        server_proc.terminate()
        for proc in edge_procs + client_procs:
            proc.terminate()
        
        print(f"✅ {scenario_name} - Seed {seed} completo")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/scenarios.yaml")
    args = parser.parse_args()
    
    orchestrator = ExperimentOrchestrator(args.config)
    orchestrator.run_experiments()
