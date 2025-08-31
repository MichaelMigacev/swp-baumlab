**Genereller Ablauf:**

1. Nutze **preprocessing.py** um 1/3 der Normalisierungsmethoden zu berechnen (Achtung nur 1 Fold)  
   Hierfür braucht man: **X.p.gz** aus dem Paper  

   **Ergebnis:**  
   **test%dval%dnorm%s.p.gz** (mit Platzhaltern für Fold und Method) → der fertig gepreprocesste Datensatz  

2. Dann habe ich einige Files erstellt, um das Modell zu erstellen und zu sichern.  
   - **model.py** ist das Modell selbst  
   - **model_save.py** erstellt die Datei `...model_pt` und ermöglicht den eigentlichen Aufruf mit Hilfe von `train_single_model`.  
   
   ⚠️ Das Ergebnis ist leider zu groß zum Uploaden, aber nötig für alle weiteren Schritte.  
   
   Model save nimmt die folgenden Parameter:  
   - layers  
   - learning_rate  
   - dropout  
   - early_stopping  
   - patience  
   - delta (fürs early stopping)  
   - indirekt die Normalisierungsstrategie  

3. Das wichtigste File ist **grid_search.py**.  
   Hier starten wir den eigentlichen GridSearch mit den Parametern aus dem Paper.  
   - Ergebnisse werden in `GridSearchresults` gespeichert  
   - pro Normalisierungsstrategie merken wir uns auch das beste Modell  
   - (das bringt nicht so viel, da viele Strategien fast gleiche Ergebnisse haben)  

4. Nötige **Libraries**:

   **Core scientific libraries**  
   - numpy  
   - pandas  
   - math  
   - random  
   - json  
   - os  
   - `from itertools import product`  
   - `from datetime import datetime`  

   **Serialization**  
   - pickle  
   - gzip  

   **PyTorch**  
   - torch  
   - torch.nn.functional  
   - `from torch.utils.data import TensorDataset, DataLoader`  

   **PyTorch Lightning**  
   - lightning  
   - lightning.pytorch  
   - `from lightning.pytorch.callbacks import EarlyStopping`  
   - `from lightning.pytorch.loggers import CSVLogger`  

   **Scikit-learn metrics**  
   - `from sklearn.metrics import root_mean_squared_error, mean_squared_error`  

5. Es gibt noch einige Spielereien, die ich separat erkläre. Ebenso das preprocessing.
