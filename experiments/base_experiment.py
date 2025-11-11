import json
import time
import torch
import flwr as fl
from typing import Dict, Any, List
import sys
import os

# Adiciona o caminho para importar seus módulos
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class BaseExperiment:
    """Classe base para todos os experimentos federados"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results = {}
        self.experiment_id = f"{config['category']}_{config['name']}_{int(time.time())}"
        self.start_time = time.time()
    
    def setup_environment(self):
        """Configura ambiente - sobrescreva conforme necessário"""
        print(f"🔧 Configurando experimento: {self.config['name']}")
        
    def run(self) -> Dict[str, Any]:
        """Executa o experimento - MÉTODO PRINCIPAL"""
        raise NotImplementedError("Cada experimento deve implementar run()")
    
    def save_results(self):
        """Salva resultados de forma padronizada"""
        self.results['experiment_duration'] = time.time() - self.start_time
        self.results['experiment_id'] = self.experiment_id
        
        result_data = {
            'metadata': {
                'experiment_id': self.experiment_id,
                'category': self.config['category'],
                'name': self.config['name'],
                'timestamp': time.time(),
                'duration_seconds': self.results['experiment_duration']
            },
            'config': self.config,
            'results': self.results
        }
        
        # Garante que a pasta results existe
        os.makedirs('results/raw', exist_ok=True)
        filename = f"results/raw/{self.experiment_id}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Resultados salvos: {filename}")
        return filename
    
    def calculate_communication_cost(self, parameters: List) -> float:
        """Calcula custo de comunicação em MB"""
        if parameters is None:
            return 0.0
        
        total_bytes = 0
        for param in parameters:
            if hasattr(param, 'nbytes'):
                total_bytes += param.nbytes
            else:
                total_bytes += param.size * param.itemsize
                
        return total_bytes / (1024 * 1024)  # Convert to MB