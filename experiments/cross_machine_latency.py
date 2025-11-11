import time
import statistics
from experiments.base_experiment import BaseExperiment
import subprocess
import platform

class CrossMachineLatencyExperiment(BaseExperiment):
    """Mede latência real entre 2 máquinas na rede"""
    
    def ping_host(self, host):
        """Faz ping para um host e retorna latency"""
        param = "-n" if platform.system().lower() == "windows" else "-c"
        command = ["ping", param, "1", host]
        
        try:
            output = subprocess.check_output(command, stderr=subprocess.STDOUT, universal_newlines=True)
            # Extrai tempo do ping no Windows
            if "ms" in output:
                for line in output.split('\n'):
                    if "time=" in line:
                        time_str = line.split("time=")[1].split("ms")[0]
                        return float(time_str)
        except:
            pass
        
        return None
    
    def run(self):
        machine2_ip = self.config.get('machine2_ip', '192.168.1.20')
        latencies = []
        
        print(f"📡 Medindo latência para {machine2_ip}...")
        
        for i in range(10):  # 10 medições
            latency = self.ping_host(machine2_ip)
            
            if latency is not None:
                latencies.append(latency)
                print(f"📊 Medição {i+1}: {latency:.2f} ms")
            else:
                print(f"❌ Falha na medição {i+1}")
            
            time.sleep(2)  # Espera 2 segundos entre medições
        
        if latencies:
            self.results = {
                'target_machine': machine2_ip,
                'average_latency_ms': statistics.mean(latencies),
                'latency_std_ms': statistics.stdev(latencies) if len(latencies) > 1 else 0,
                'min_latency_ms': min(latencies),
                'max_latency_ms': max(latencies),
                'packet_loss_percent': (10 - len(latencies)) / 10 * 100,
                'all_measurements': latencies
            }
        else:
            self.results = {
                'target_machine': machine2_ip,
                'error': 'Não foi possível medir latência',
                'packet_loss_percent': 100
            }
        
        print(f"✅ Latência média: {self.results.get('average_latency_ms', 'N/A')} ms")
        print(f"📉 Packet loss: {self.results.get('packet_loss_percent', 0):.1f}%")
        
        return self.save_results()

# Teste rápido
if __name__ == "__main__":
    config = {
        'category': 'network',
        'name': 'latency_test',
        'machine2_ip': '192.168.1.20'  # AJUSTE para IP da sua máquina 2
    }
    
    experiment = CrossMachineLatencyExperiment(config)
    experiment.run()