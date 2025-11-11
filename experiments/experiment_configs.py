"""
Configurações para os experimentos com 2 máquinas
"""

# Configurações base para 2 máquinas
BASE_CONFIG_2_MACHINES = {
    "dataset": "CIFAR10",
    "normal_class": 0,
    "use_transfer_learning": True,
    "total_clients": 10,
    "min_clients": 3,
    "scenario": "small"
}

# Experimentos otimizados para 2 máquinas
TWO_MACHINE_EXPERIMENTS = [
    {
        "category": "efficiency",
        "name": "cross_machine_communication",
        "description": "Comunicação entre 2 máquinas reais",
        "duration_minutes": 5,
        "machine1_roles": ["cloud", "fog1", "edge1"],
        "machine2_roles": ["fog2", "edge2"]
    },
    {
        "category": "robustness", 
        "name": "natural_heterogeneity",
        "description": "Heterogeneidade natural entre 2 máquinas diferentes",
        "duration_minutes": 7,
        "machine1_roles": ["cloud", "fog1", "edge1"],
        "machine2_roles": ["fog2", "edge2"],
        "straggler_percentage": 0.2
    },
    {
        "category": "network",
        "name": "real_latency_impact", 
        "description": "Impacto da latência real de rede",
        "duration_minutes": 6,
        "measure_latency": True
    }
]

def get_2_machine_config(experiment_name):
    """Retorna configuração para experimento com 2 máquinas"""
    for exp in TWO_MACHINE_EXPERIMENTS:
        if exp['name'] == experiment_name:
            return {**BASE_CONFIG_2_MACHINES, **exp}
    return None