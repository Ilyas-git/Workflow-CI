# Workflow-CI (MLflow + Scikit-Learn)

Struktur ini menyiapkan:
- MLflow Project (folder `MLProject/`) untuk menjalankan training via `mlflow run`
- GitHub Actions workflow untuk menjalankan training saat trigger `workflow_dispatch` atau push ke branch `main/master`

## Menjalankan secara lokal
```bash
cd MLProject
mlflow run . -e main -P tracking_dir=mlruns_local
mlflow ui --backend-store-uri ./mlruns_local --host 127.0.0.1 --port 5000
```

## Dataset
Folder `MLProject/namadataset_preprocessing/` berisi CSV contoh agar workflow CI dapat berjalan tanpa kredensial Kaggle.
Silakan ganti file tersebut dengan hasil preprocessing Anda (output dari `Eksperimen.ipynb`).
