import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

class StatisticalAnalyzer:
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.results = {}
    
    def generate_comprehensive_report(self, experiment_results: Dict) -> Dict:
        """Gera relatório estatístico completo"""
        report = {
            'descriptive_statistics': {},
            'hypothesis_testing': {},
            'correlation_analysis': {},
            'effect_sizes': {},
            'anova_results': {},
            'power_analysis': {}
        }
        
        # Análise descritiva
        report['descriptive_statistics'] = self.calculate_descriptive_stats(experiment_results)
        
        # Testes de hipótese
        report['hypothesis_testing'] = self.perform_hypothesis_tests(experiment_results)
        
        # Análise de correlação
        report['correlation_analysis'] = self.analyze_correlations(experiment_results)
        
        # ANOVA entre cenários
        report['anova_results'] = self.perform_anova(experiment_results)
        
        # Análise de poder estatístico
        report['power_analysis'] = self.power_analysis(experiment_results)
        
        return report
    
    def calculate_descriptive_stats(self, experiment_results: Dict) -> Dict:
        """Calcula estatísticas descritivas para cada cenário"""
        descriptive_stats = {}
        
        for scenario_name, scenario_runs in experiment_results.items():
            accuracies = []
            f1_scores = []
            training_times = []
            communication_costs = []
            
            for run in scenario_runs:
                metrics = run.get('metrics', {})
                accuracies.append(metrics.get('accuracy', 0))
                f1_scores.append(metrics.get('f1_score', 0))
                training_times.append(run.get('duration', 0))
                communication_costs.append(metrics.get('total_communication_bytes', 0) / (1024 * 1024))  # MB
            
            descriptive_stats[scenario_name] = {
                'accuracy': {
                    'mean': float(np.mean(accuracies)),
                    'median': float(np.median(accuracies)),
                    'std': float(np.std(accuracies)),
                    'min': float(np.min(accuracies)),
                    'max': float(np.max(accuracies)),
                    'ci_95': self.calculate_confidence_interval(accuracies),
                    'cv': float(np.std(accuracies) / np.mean(accuracies)) if np.mean(accuracies) > 0 else 0
                },
                'f1_score': {
                    'mean': float(np.mean(f1_scores)),
                    'std': float(np.std(f1_scores)),
                    'ci_95': self.calculate_confidence_interval(f1_scores)
                },
                'training_time_hours': {
                    'mean': float(np.mean(training_times) / 3600),
                    'total': float(sum(training_times) / 3600)
                },
                'communication_mb': {
                    'mean': float(np.mean(communication_costs)),
                    'total': float(sum(communication_costs))
                }
            }
        
        return descriptive_stats
    
    def perform_hypothesis_tests(self, experiment_results: Dict) -> Dict:
        """Realiza testes de hipótese entre diferentes cenários"""
        hypothesis_tests = {}
        scenarios = list(experiment_results.keys())
        
        # Compara todos os pares de cenários
        for i, scenario_a in enumerate(scenarios):
            for scenario_b in scenarios[i+1:]:
                comparison_key = f"{scenario_a}_vs_{scenario_b}"
                
                # Extrai acurácias
                accuracies_a = [run['metrics'].get('accuracy', 0) for run in experiment_results[scenario_a]]
                accuracies_b = [run['metrics'].get('accuracy', 0) for run in experiment_results[scenario_b]]
                
                # Teste t de Student
                t_stat, p_value_ttest = stats.ttest_ind(accuracies_a, accuracies_b)
                
                # Teste de Mann-Whitney (não paramétrico)
                u_stat, p_value_mw = stats.mannwhitneyu(accuracies_a, accuracies_b)
                
                # Teste de Wilcoxon
                w_stat, p_value_wilcoxon = stats.ranksums(accuracies_a, accuracies_b)
                
                hypothesis_tests[comparison_key] = {
                    't_test': {
                        'statistic': float(t_stat),
                        'p_value': float(p_value_ttest),
                        'significant': p_value_ttest < self.significance_level
                    },
                    'mann_whitney': {
                        'statistic': float(u_stat),
                        'p_value': float(p_value_mw),
                        'significant': p_value_mw < self.significance_level
                    },
                    'wilcoxon': {
                        'statistic': float(w_stat),
                        'p_value': float(p_value_wilcoxon),
                        'significant': p_value_wilcoxon < self.significance_level
                    },
                    'effect_size': self.calculate_effect_size(accuracies_a, accuracies_b)
                }
        
        return hypothesis_tests
    
    def analyze_correlations(self, experiment_results: Dict) -> Dict:
        """Analisa correlações entre métricas"""
        all_data = []
        
        for scenario_name, scenario_runs in experiment_results.items():
            for run in scenario_runs:
                metrics = run.get('metrics', {})
                all_data.append({
                    'scenario': scenario_name,
                    'accuracy': metrics.get('accuracy', 0),
                    'f1_score': metrics.get('f1_score', 0),
                    'training_time': run.get('duration', 0),
                    'communication_mb': metrics.get('total_communication_bytes', 0) / (1024 * 1024),
                    'client_count': len([k for k in run.keys() if 'client' in k])
                })
        
        df = pd.DataFrame(all_data)
        
        # Matriz de correlação
        numeric_cols = ['accuracy', 'f1_score', 'training_time', 'communication_mb', 'client_count']
        correlation_matrix = df[numeric_cols].corr()
        
        # Testes de significância de correlações
        correlation_tests = {}
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                corr_coef, p_value = stats.pearsonr(df[col1], df[col2])
                correlation_tests[f"{col1}_vs_{col2}"] = {
                    'pearson_r': float(corr_coef),
                    'p_value': float(p_value),
                    'significant': p_value < self.significance_level
                }
        
        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'correlation_tests': correlation_tests,
            'summary': f"Analisadas {len(df)} observações entre {len(numeric_cols)} variáveis"
        }
    
    def perform_anova(self, experiment_results: Dict) -> Dict:
        """ANOVA para comparar múltiplos cenários"""
        anova_data = []
        group_labels = []
        
        for scenario_name, scenario_runs in experiment_results.items():
            accuracies = [run['metrics'].get('accuracy', 0) for run in scenario_runs]
            anova_data.extend(accuracies)
            group_labels.extend([scenario_name] * len(accuracies))
        
        if len(set(group_labels)) < 2:
            return {'error': 'ANOVA requer pelo menos 2 grupos'}
        
        # ANOVA unidirecional
        f_stat, p_value = stats.f_oneway(*[anova_data for scenario in experiment_results.keys()])
        
        # Teste post-hoc de Tukey
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        tukey_result = pairwise_tukeyhsd(anova_data, group_labels, alpha=self.significance_level)
        
        return {
            'anova': {
                'f_statistic': float(f_stat),
                'p_value': float(p_value),
                'significant': p_value < self.significance_level
            },
            'tukey_hsd': {
                'summary': str(tukey_result.summary()),
                'reject': tukey_result.reject.tolist(),
                'groups': tukey_result.groups,
                'means': [np.mean([run['metrics'].get('accuracy', 0) for run in runs]) 
                         for runs in experiment_results.values()]
            }
        }
    
    def power_analysis(self, experiment_results: Dict) -> Dict:
        """Análise de poder estatístico"""
        power_analysis = {}
        
        for scenario_name, scenario_runs in experiment_results.items():
            accuracies = [run['metrics'].get('accuracy', 0) for run in scenario_runs]
            effect_size = np.std(accuracies) / np.mean(accuracies) if np.mean(accuracies) > 0 else 0
            
            # Poder do teste para detectar diferenças
            from statsmodels.stats.power import TTestIndPower
            power_analyzer = TTestIndPower()
            power = power_analyzer.solve_power(
                effect_size=effect_size,
                nobs=len(accuracies),
                alpha=self.significance_level
            )
            
            power_analysis[scenario_name] = {
                'effect_size': float(effect_size),
                'statistical_power': float(power),
                'sample_size': len(accuracies),
                'min_detectable_effect': self.calculate_mde(accuracies, power)
            }
        
        return power_analysis
    
    def calculate_confidence_interval(self, data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Calcula intervalo de confiança"""
        if len(data) < 2:
            return (0.0, 0.0)
        
        mean = np.mean(data)
        sem = stats.sem(data)
        ci = stats.t.interval(confidence, len(data)-1, loc=mean, scale=sem)
        return (float(ci[0]), float(ci[1]))
    
    def calculate_effect_size(self, group_a: List[float], group_b: List[float]) -> Dict:
        """Calcula tamanhos de efeito"""
        mean_a, mean_b = np.mean(group_a), np.mean(group_b)
        std_a, std_b = np.std(group_a), np.std(group_b)
        n_a, n_b = len(group_a), len(group_b)
        
        # Cohen's d
        pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
        cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0
        
        # Hedges' g (corrigido para pequenas amostras)
        hedges_g = cohens_d * (1 - (3 / (4 * (n_a + n_b) - 9)))
        
        return {
            'cohens_d': float(cohens_d),
            'hedges_g': float(hedges_g),
            'glass_delta': (mean_a - mean_b) / std_b if std_b > 0 else 0,
            'interpretation': self.interpret_effect_size(cohens_d)
        }
    
    def interpret_effect_size(self, effect_size: float) -> str:
        """Interpreta o tamanho do efeito"""
        abs_effect = abs(effect_size)
        if abs_effect < 0.2:
            return "Trivial"
        elif abs_effect < 0.5:
            return "Pequeno"
        elif abs_effect < 0.8:
            return "Médio"
        else:
            return "Grande"
    
    def calculate_mde(self, data: List[float], power: float = 0.8) -> float:
        """Calcula Minimum Detectable Effect"""
        from statsmodels.stats.power import TTestIndPower
        power_analyzer = TTestIndPower()
        
        mde = power_analyzer.solve_power(
            effect_size=None,
            nobs=len(data),
            alpha=self.significance_level,
            power=power
        )
        
        return float(mde) if mde else 0.0
    
    def generate_visualizations(self, experiment_results: Dict, output_dir: str = "results/plots"):
        """Gera visualizações estatísticas"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Box plot comparativo
        plt.figure(figsize=(12, 8))
        data_to_plot = []
        labels = []
        
        for scenario_name, scenario_runs in experiment_results.items():
            accuracies = [run['metrics'].get('accuracy', 0) for run in scenario_runs]
            data_to_plot.append(accuracies)
            labels.append(scenario_name)
        
        plt.boxplot(data_to_plot, labels=labels)
        plt.title('Distribuição de Acurácias por Cenário')
        plt.ylabel('Acurácia')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/accuracy_boxplot.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Heatmap de correlações
        plt.figure(figsize=(10, 8))
        all_data = []
        
        for scenario_name, scenario_runs in experiment_results.items():
            for run in scenario_runs:
                metrics = run.get('metrics', {})
                all_data.append({
                    'accuracy': metrics.get('accuracy', 0),
                    'f1': metrics.get('f1_score', 0),
                    'time': run.get('duration', 0) / 3600,
                    'communication': metrics.get('total_communication_bytes', 0) / (1024 * 1024)
                })
        
        df = pd.DataFrame(all_data)
        corr_matrix = df.corr()
        
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
        plt.title('Matriz de Correlação entre Métricas')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/correlation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()