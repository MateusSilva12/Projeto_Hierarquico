import subprocess
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class TwoMachineExperiment:
    def __init__(self, machine1_ip, machine2_ip):
        self.machine1_ip = machine1_ip
        self.machine2_ip = machine2_ip
        self.processes = []
    
    def run_on_machine1(self):
        """Executa Cloud + Fog1 + Edge1 na Máquina 1"""
        print("🚀 Iniciando serviços na Máquina 1...")
        
        # Servidor Global (Cloud) - PORTA 9080
        cloud_cmd = [
            sys.executable, 'servers/global_server_anomaly.py',
            '--rounds', '10',
            '--min-clients', '2',
            '--architecture', 'hierarchical',
            '--scenario', 'small',
            '--port', '9080'
        ]
        
        # Fog Server 1 - PORTA 9081
        fog1_cmd = [
            sys.executable, 'servers/fog_server.py',
            '--fog-port', '9081',
            '--cloud-ip', f'{self.machine1_ip}:9080',
            '--edge-ports', '9082',
            '--scenario', 'small'
        ]
        
        # Edge Aggregator 1 - PORTA 9082
        edge1_cmd = [
            sys.executable, 'servers/aggregator_anomaly.py',
            '--port', '9082',
            '--server-ip', f'{self.machine1_ip}:9081',
            '--min-clients', '3'
        ]
        
        # Inicia processos
        processes = []
        print("🚀 SERVIDOR GLOBAL INICIADO")
        processes.append(subprocess.Popen(cloud_cmd))
        time.sleep(8)
        
        print("🌫️  FOG SERVER: Porta 9081")
        processes.append(subprocess.Popen(fog1_cmd))
        time.sleep(5)
        
        print("🏢 Agregador: Iniciando em 0.0.0.0:9082")
        processes.append(subprocess.Popen(edge1_cmd))
        time.sleep(3)
        
        return processes
    
    def run_on_machine2(self):
        """Executa Fog2 + Edge2 na Máquina 2"""
        print("🚀 Iniciando serviços na Máquina 2...")
        
        # Fog Server 2 - PORTA 9083
        fog2_cmd = [
            sys.executable, 'servers/fog_server.py',
            '--fog-port', '9083', 
            '--cloud-ip', f'{self.machine1_ip}:9080',
            '--edge-ports', '9084',
            '--scenario', 'small'
        ]
        
        # Edge Aggregator 2 - PORTA 9084
        edge2_cmd = [
            sys.executable, 'servers/aggregator_anomaly.py',
            '--port', '9084',
            '--server-ip', f'{self.machine2_ip}:9083',
            '--min-clients', '3'
        ]
        
        processes = []
        print("🌫️  FOG SERVER: Porta 9083")
        processes.append(subprocess.Popen(fog2_cmd))
        time.sleep(5)
        
        print("🏢 Agregador: Iniciando em 0.0.0.0:9084")
        processes.append(subprocess.Popen(edge2_cmd))
        time.sleep(3)
        
        return processes
    
    def start_clients(self):
        """Inicia clientes distribuídos entre as máquinas"""
        print("👥 Iniciando clientes...")
        
        client_processes = []
        
        # Clientes para Edge1 (Máquina 1) - 5 clientes
        for i in range(5):
            profile = 'medium' if i % 2 == 0 else 'low_end'
            memory = '4 GB' if profile == 'medium' else '2 GB'
            print(f"ℹ️  Cliente {i}: Memória alvo: {memory}")
            
            cmd = [
                sys.executable, 'cliente_anomaly_sgd.py',
                '--server-ip', f'{self.machine1_ip}:9082',
                '--client-id', str(i),
                '--profile', profile,
                '--scenario', 'small'
            ]
            client_processes.append(subprocess.Popen(cmd))
            time.sleep(2)
        
        # Clientes para Edge2 (Máquina 2) - 5 clientes  
        for i in range(5, 10):
            profile = 'high_end' if i == 7 else 'medium'
            memory = '8 GB' if profile == 'high_end' else '4 GB'
            print(f"ℹ️  Cliente {i}: Memória alvo: {memory}")
            
            cmd = [
                sys.executable, 'cliente_anomaly_sgd.py',
                '--server-ip', f'{self.machine2_ip}:9084',
                '--client-id', str(i),
                '--profile', profile,
                '--scenario', 'small'
            ]
            client_processes.append(subprocess.Popen(cmd))
            time.sleep(2)
        
        return client_processes
    
    def cleanup(self, processes):
        """Limpeza adequada dos processos"""
        for p in processes:
            try:
                p.terminate()
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
            except:
                pass
    
    def run_experiment(self, experiment_name, duration_minutes=3):
        """Executa um experimento completo"""
        print(f"🔬 INICIANDO EXPERIMENTO: {experiment_name}")
        print(f"⏰ Duração estimada: {duration_minutes} minutos")
        
        all_processes = []
        
        try:
            # 1. Inicia servidores na Máquina 1
            m1_processes = self.run_on_machine1()
            all_processes.extend(m1_processes)
            time.sleep(10)
            
            # 2. Inicia servidores na Máquina 2  
            m2_processes = self.run_on_machine2()
            all_processes.extend(m2_processes)
            time.sleep(10)
            
            # 3. Inicia clientes
            clients = self.start_clients()
            all_processes.extend(clients)
            
            # 4. Aguarda experimento rodar
            print(f"⏳ Experimento {experiment_name} em execução...")
            time.sleep(duration_minutes * 60)
            
        except Exception as e:
            print(f"❌ Erro no experimento: {e}")
        finally:
            # 5. Finaliza processos
            print("🛑 Finalizando experimento...")
            self.cleanup(all_processes)
            print(f"✅ Experimento {experiment_name} concluído!")

# Configuração para suas máquinas
if __name__ == "__main__":
    orchestrator = TwoMachineExperiment(
        machine1_ip="192.168.1.9",   # Máquina principal
        machine2_ip="192.168.1.21"   # Máquina secundária
    )
    
    orchestrator.run_experiment("teste_2_maquinas", duration_minutes=3)