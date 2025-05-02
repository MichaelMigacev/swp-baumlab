
import requests
import zipfile
import io
from pathlib import Path

# Zielverzeichnis für entpackte Daten
out_dir = Path("data")
out_dir.mkdir(exist_ok=True)

# URL zur ZIP-Datei mit den DeepSynergy-Daten
url = "http://www.bioinf.jku.at/software/DeepSynergy/data.zip"

print("📦 Lade ZIP-Datei herunter von:", url)

# Lade ZIP-Datei über HTTP
response = requests.get(url)

# Prüfen ob der Download geklappt hat
if response.status_code != 200:
    print("❌ Fehler beim Herunterladen:", response.status_code)
    exit()

# Entpacke ZIP-Inhalt im Arbeitsspeicher
zip_data = zipfile.ZipFile(io.BytesIO(response.content))

print("📂 Entpacke Dateien nach 'data/' ...")
zip_data.extractall(out_dir)

print("✅ Download und Entpacken abgeschlossen.")
print("📁 Dateien gespeichert in:", out_dir.resolve())
