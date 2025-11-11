# corrigir_flower.py
import subprocess
import sys

def corrigir_flower():
    print("🔧 CORRIGINDO INSTALAÇÃO DO FLOWER...")
    
    # Lista pacotes problemáticos
    pacotes = ["flwr", "flower", "grpcio", "grpcio-tools", "protobuf"]
    
    for pacote in pacotes:
        print(f"🧹 Removendo {pacote}...")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", pacote])
    
    print("📦 Instalando Flower limpo...")
    result = subprocess.run([
        sys.executable, "-m", "pip", "install", "flwr==1.5.0", 
        "grpcio==1.60.0", "protobuf==4.25.3", "numpy"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Instalação bem-sucedida!")
    else:
        print("❌ Erro na instalação, tentando alternativa...")
        subprocess.run([sys.executable, "-m", "pip", "install", "flwr", "--no-deps"])
        subprocess.run([sys.executable, "-m", "pip", "install", "grpcio", "protobuf", "numpy"])
    
    # Testar
    print("\n🧪 Testando importação...")
    try:
        import flwr
        print(f"✅ FLOWER FUNCIONANDO! Versão: {flwr.__version__}")
        print("📁 Atributos disponíveis:")
        for attr in [x for x in dir(flwr) if not x.startswith('_')]:
            print(f"   - {attr}")
    except Exception as e:
        print(f"❌ Erro: {e}")

if __name__ == "__main__":
    corrigir_flower()