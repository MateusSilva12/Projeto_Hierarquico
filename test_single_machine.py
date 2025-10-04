# test_2_notebooks.py
import subprocess
import time
import os
import sys

def get_ip_address():
    """Pega o IP da máquina (simplificado)"""
    return "127.0.0.1"  # Mude para o IP real do notebook

def test_distributed_setup():
    """Testa configuração distribuída entre 2 notebooks"""
    print("🖥️🧪 TESTE PARA 2 NOTEBOOKS")
    print("=" * 50)
    
    notebook1_ip = input("📡 IP do Notebook 1 (Servidor): ").strip() or "127.0.0.1"
    notebook2_ip = input("📡 IP do Notebook 2 (Clientes): ").strip() or "127.0.0.1"
    
    print(f"\n🎯 Configuração:")
    print(f"   Notebook 1 (Servidor): {notebook1_ip}:8080")
    print(f"   Notebook 2 (Clientes): {notebook2_ip}:8040")
    
    choice = input("\nEste notebook é o [1] Servidor ou [2] Cliente? ")
    
    if choice == "1":
        run_server(notebook1_ip)
    elif choice == "2":
        run_clients(notebook2_ip, notebook1_ip)
    else:
        print("❌ Escolha inválida")

def run_server(server_ip):
    """Executa no Notebook 1 (Servidor)"""
    print(f"\n🚀 INICIANDO SERVIDOR em {server_ip}:8080")
    
    server_code = f"""
import flwr as fl
import time

print("🌐 Servidor distribuído iniciando...")

strategy = fl.server.strategy.FedAvg(
    min_available_clients=2,
    min_fit_clients=2,
    fraction_fit=1.0
)

fl.server.start_server(
    server_address="0.0.0.0:8080",  # Ouça em todas as interfaces
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy
)
print("✅ Servidor finalizado!")
"""
    
    with open("distributed_server.py", "w") as f:
        f.write(server_code)
    
    print("⏳ Iniciando servidor...")
    subprocess.run([sys.executable, "distributed_server.py"])
    
    # Limpeza
    if os.path.exists("distributed_server.py"):
        os.remove("distributed_server.py")

def run_clients(client_ip, server_ip):
    """Executa no Notebook 2 (Clientes)"""
    print(f"\n👥 INICIANDO CLIENTES em {client_ip}")
    print(f"   Conectando ao servidor: {server_ip}:8080")
    
    processes = []
    
    try:
        # Inicia 2 clientes
        for i in range(2):
            client_code = f"""
import flwr as fl
import numpy as np
import time

class DistributedClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [np.ones(10, dtype=np.float32)]
    
    def fit(self, parameters, config):
        print(f"Cliente {i} (Notebook 2): Treinando...")
        time.sleep(2)
        return [np.ones(10, dtype=np.float32)], 10, {{"loss": 0.1}}
    
    def evaluate(self, parameters, config):
        return 0.1, 10, {{"accuracy": 0.9}}

print(f"👤 Cliente {i} do Notebook 2 conectando...")
fl.client.start_numpy_client(
    server_address="{server_ip}:8080",  # Conecta ao Notebook 1
    client=DistributedClient()
)
print(f"✅ Cliente {i} finalizado!")
"""
            filename = f"distributed_client_{i}.py"
            with open(filename, "w") as f:
                f.write(client_code)
            
            proc = subprocess.Popen([sys.executable, filename])
            processes.append(proc)
            time.sleep(1)
        
        # Aguarda
        print("⏳ Aguardando execução (30 segundos)...")
        time.sleep(30)
        
    finally:
        # Cleanup
        print("🧹 Limpando...")
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
        
        for i in range(2):
            filename = f"distributed_client_{i}.py"
            if os.path.exists(filename):
                os.remove(filename)
        
        print("🏁 Clientes finalizados!")

if __name__ == "__main__":
    test_distributed_setup()