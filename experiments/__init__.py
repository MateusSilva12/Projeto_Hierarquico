"""
Módulo de experimentos para aprendizagem federada hierárquica
"""

from .base_experiment import BaseExperiment
from .two_machine_orchestrator import TwoMachineExperiment
from .cross_machine_latency import CrossMachineLatencyExperiment

__all__ = [
    'BaseExperiment',
    'TwoMachineExperiment', 
    'CrossMachineLatencyExperiment'
]