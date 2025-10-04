import torch
import numpy as np
from collections import OrderedDict

def get_parameters(net) -> list[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: list[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def calculate_anomaly_score(model, images, threshold=0.1):
    """Calcula score de anomalia baseado no erro de reconstrução"""
    model.eval()
    with torch.no_grad():
        _, reconstructions = model(images)
        reconstruction_error = torch.mean(torch.pow(reconstructions - images, 2), dim=[1,2,3])
        return reconstruction_error > threshold