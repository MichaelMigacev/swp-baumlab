import torch
from model import DeepSynergyModel

input_size = 8846
model = DeepSynergyModel(input_size)
model.load_state_dict(torch.load("deepsynergy_model.pt"))
model.eval()

print("Modell erfolgreich geladen und bereit für Vorhersagen")
