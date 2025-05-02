## Weekly Goals and TODOs

##### This contains our weekly goals as well as the tracked worked on tasks

### Week 2

#### Goals: 
DeepSynergy:
- rewrite DeepSynergy to pytorch
- Understand input for DS (DeepSynergy) mostly (with biological details) and present to group
DeepSignalingFlow:
- Understand input for DSF (DeepSignalingFlow) mostly (with biological details) and present to group
- Understanding all the operations of DFS with corresponding code pieces and present to group
  - copy DSF elements to our github
Both:
- understand validation standard (leave blablabla out) for data leakage free validation
  - test with DS

#### Table Week 2

| Names | Tasks | Time Taken |
|----------|----------|----------|
| Michael F   | Cell 2   | Cell 5   |
| Sebastian | DeepSynergy von TensorFlow nach PyTorch übertragen
→ Architektur (model.py) manuell in PyTorch mit nn.Sequential nachgebaut.
Datensatz verarbeitet & geladen
→ data_test_fold0_tanh.p erfolgreich mit pickle eingelesen, in TensorDataset umgewandelt.
Trainings-Pipeline in PyTorch erstellt & ausgeführt
→ train.py mit Trainings- und Validierungsschleife programmiert und mit realen Daten (über Stunden) trainiert.
Data-Leakage-Prüfsystem entwickelt
→ check_data_leakage.py erstellt, AB/BA-Kombinationen identifiziert (zunächst auf Testdatei, dann vorbereitet für echte Daten).
Struktur zur Wiederverwendbarkeit vorbereitet
→ Modell sollte per torch.save gesichert, auf GitHub geladen und mit load_model.py nachnutzbar werden.   |  Cell 10  |
| Olha | Cell 12  |  Cell 15  |
| Zhao  | Cell 17  |  Cell 20  |
| Michael M  | Organized Repo  | Cell 25  |

