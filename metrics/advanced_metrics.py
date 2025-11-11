import numpy as np
import json
import pandas as pd
from typing import Dict, List, Tuple
from scipy import stats
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class AdvancedMetrics:
    def __init__(self):
        self.metrics_history = {}
    
    def calculate_comprehensive_metrics(self, y_true: List, y_pred: List, y_scores: List) -> Dict:
        """Calcula métricas abrangentes de classificação"""
        accuracy = np.mean(np.array(y_true) == np.array(y_pred))
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        
        try:
            auc = roc_auc_score(y_true, y_scores)
        except:
            auc = 0.0
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc': float(auc),
            'confusion_matrix': cm.tolist(),
            'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            'false_positive_rate': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        }
    
    def communication_efficiency(self, accuracy: float, total_bytes: int) -> float:
        """Calcula eficiência de comunicação (acurácia por MB)"""
        mb_transferred = total_bytes / (1024 * 1024)
        return accuracy / mb_transferred if mb_transferred > 0 else 0.0
    
    def convergence_analysis(self, accuracy_history: List[float], threshold: float = 0.95) -> Dict:
        """Analisa velocidade e qualidade da convergência"""
        if not accuracy_history:
            return {'converged': False, 'rounds_to_converge': None, 'final_accuracy': 0.0}
        
        max_accuracy = max(accuracy_history)
        target_accuracy = max_accuracy * threshold
        
        convergence_round = None
        for round_num, acc in enumerate(accuracy_history):
            if acc >= target_accuracy:
                convergence_round = round_num + 1
                break
        
        return {
            'converged': convergence_round is not None,
            'rounds_to_converge': convergence_round,
            'final_accuracy': float(accuracy_history[-1]),
            'max_accuracy': float(max_accuracy),
            'convergence_speed': convergence_round if convergence_round else len(accuracy_history)
        }
    
    def statistical_significance_test(self, method_a_scores: List[float], 
                                    method_b_scores: List[float]) -> Dict:
        """Teste de significância estatística entre dois métodos"""
        if len(method_a_scores) < 2 or len(method_b_scores) < 2:
            return {'significant': False, 'p_value': 1.0, 'effect_size': 0.0}
        
        # Teste t de Student
        t_stat, p_value_ttest = stats.ttest_ind(method_a_scores, method_b_scores)
        
        # Teste de Wilcoxon (não paramétrico)
        try:
            wilcoxon_stat, p_value_wilcoxon = stats.ranksums(method_a_scores, method_b_scores)
        except:
            p_value_wilcoxon = 1.0
        
        # Tamanho do efeito
        mean_a, mean_b = np.mean(method_a_scores), np.mean(method_b_scores)
        std_pooled = np.sqrt((np.var(method_a_scores) + np.var(method_b_scores)) / 2)
        effect_size = (mean_a - mean_b) / std_pooled if std_pooled > 0 else 0.0
        
        return {
            't_test_p_value': float(p_value_ttest),
            'wilcoxon_p_value': float(p_value_wilcoxon),
            'significant_ttest': p_value_ttest < 0.05,
            'significant_wilcoxon': p_value_wilcoxon < 0.05,
            'effect_size': float(effect_size),
            'mean_difference': float(mean_a - mean_b),
            'confidence_interval': stats.t.interval(0.95, 
                len(method_a_scores)+len(method_b_scores)-2,
                loc=mean_a-mean_b,
                scale=std_pooled*np.sqrt(1/len(method_a_scores) + 1/len(method_b_scores)))
        }
    
    def cost_analysis(self, training_times: List[float], 
                     communication_costs: List[float],
                     accuracy: float) -> Dict:
        """Análise de custo-benefício do treinamento"""
        total_training_time = sum(training_times)
        total_communication = sum(communication_costs) / (1024 * 1024)  # MB
        
        return {
            'total_training_time_hours': total_training_time / 3600,
            'total_communication_mb': total_communication,
            'accuracy_per_hour': accuracy / (total_training_time / 3600) if total_training_time > 0 else 0,
            'accuracy_per_mb': accuracy / total_communication if total_communication > 0 else 0,
            'cost_efficiency': accuracy / (total_training_time / 3600 + total_communication / 1000)
        }
    
    def generate_metrics_report(self, experiment_results: Dict) -> Dict:
        """Gera relatório completo de métricas"""
        report = {
            'scenario_analysis': {},
            'statistical_comparisons': {},
            'convergence_analysis': {},
            'cost_analysis': {},
            'hardware_impact': {}
        }
        
        for scenario_name, scenario_runs in experiment_results.items():
            scenario_metrics = self.analyze_scenario(scenario_runs)
            report['scenario_analysis'][scenario_name] = scenario_metrics
            
            # Análise de convergência
            accuracies = [run['metrics'].get('final_accuracy', 0) for run in scenario_runs]
            report['convergence_analysis'][scenario_name] = self.convergence_analysis(accuracies)
        
        # Comparações estatísticas entre cenários
        report['statistical_comparisons'] = self.compare_scenarios(experiment_results)
        
        return report
    
    def analyze_scenario(self, scenario_runs: List[Dict]) -> Dict:
        """Analisa múltiplas execuções de um cenário"""
        all_accuracies = []
        all_f1_scores = []
        all_auc_scores = []
        all_training_times = []
        all_communication_costs = []
        
        for run in scenario_runs:
            metrics = run.get('metrics', {})
            all_accuracies.append(metrics.get('accuracy', 0))
            all_f1_scores.append(metrics.get('f1_score', 0))
            all_auc_scores.append(metrics.get('auc', 0))
            all_training_times.append(run.get('duration', 0))
            all_communication_costs.append(metrics.get('total_communication_bytes', 0))
        
        return {
            'accuracy': {
                'mean': float(np.mean(all_accuracies)),
                'std': float(np.std(all_accuracies)),
                'ci_95': stats.t.interval(0.95, len(all_accuracies)-1, 
                                        loc=np.mean(all_accuracies), 
                                        scale=stats.sem(all_accuracies))
            },
            'f1_score': {
                'mean': float(np.mean(all_f1_scores)),
                'std': float(np.std(all_f1_scores))
            },
            'auc': {
                'mean': float(np.mean(all_auc_scores)),
                'std': float(np.std(all_auc_scores))
            },
            'training_time': {
                'total_hours': float(sum(all_training_times) / 3600),
                'mean_per_run': float(np.mean(all_training_times))
            },
            'communication': {
                'total_mb': float(sum(all_communication_costs) / (1024 * 1024)),
                'mean_per_run': float(np.mean(all_communication_costs) / (1024 * 1024))
            }
        }