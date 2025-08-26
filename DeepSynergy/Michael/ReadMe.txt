Genereller Ablauf:

Nutze preprocessing.py um 1/3 der Normalisierungsmethoden zu berechnen (Achtung nur 1 Fold) Hierfür braucht man: X.p.gz aus dem Paper
Ergebnis: test%dval%dnorm%s.p.gz (Hier mit Platzhaltern für Fold und Method) der fertig gepreprocesste Datensatz

Dann habe ich einige Files erstellt, um das Modell zu erstellen und zu sichern. model.py ist dabei das Modell selbst, während model_save.py dazu dient die Datei ...model_pt zu erstellen und den eigentlichen Aufruf mit Hilfe von train_single_model ermöglicht. Das Ergebnis ist leider zu groß zum uploaden aber nötig für alle weiteren Schritte. Model save nimmt dabei die folgenden Parameter: layers, learning_rate, dropout, early_stopping, patience und delta (fürs early stopping) und indirekt die Normalisierungsstrategie

Das wichtigste File ist grid_search.py. Hier starten wir den eigentlich GridSearch mit den Parametern aus dem Paper. Die Ergebnisse speichern wir in GridSearchresults und merken uns pro Normalisierungsstrategie auch das beste Modell (Wobei das nicht so viel bringt, da viele Strategien fast gleiches Ergebnis haben).

Nötige Libraries:

Core scientific libraries numpy, pandas, math, random, json, os, from itertools import product, from datetime import datetime

Serialization pickle, gzip

PyTorch torch, torch.nn.functional, from torch.utils.data import TensorDataset, DataLoader

PyTorch Lightning lightning, lightning.pytorch, from lightning.pytorch.callbacks import EarlyStopping, from lightning.pytorch.loggers import CSVLogger

Scikit-learn metrics from sklearn.metrics import root_mean_squared_error, mean_squared_error

Es gibt noch einige Spielerein die ich separat erkläre. Ebenso das preprocessing.
