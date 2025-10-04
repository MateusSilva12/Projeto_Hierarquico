# setup_directories.py
import os

def create_directory_structure():
    """Cria estrutura de diretórios necessária"""
    directories = [
        "configs",
        "metrics", 
        "results",
        "servers",
        "core"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Criado: {directory}/")
    
    # Move arquivos para diretórios corretos (se necessário)
    file_mappings = {
        "global_server_anomaly.py": "servers/",
        "aggregator_anomaly.py": "servers/", 
        "fog_server.py": "servers/",
        "cliente_anomaly_sgd.py": "core/",
        "model_anomaly.py": "core/",
        "dataset_anomaly.py": "core/",
        "utils_anomaly.py": "core/"
    }
    
    print("✅ Estrutura de diretórios criada!")

if __name__ == "__main__":
    create_directory_structure()