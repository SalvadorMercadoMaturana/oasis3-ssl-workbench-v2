# OASIS3 SSL Workbench

¿Aprende un encoder **autosupervisado** (SSL), viendo sólo resonancias magnéticas sin
etiquetas, algo que distinga Alzheimer (AD) de sujetos sanos (HC)? Este repositorio entrena
encoders 3D con VICReg, BYOL, DINOv2 y tareas de pretexto sobre cubos del hipocampo de
OASIS3, los congela, y mide cuánto rinde un clasificador lineal (el *probe*) sobre sus
embeddings, frente al mismo encoder sin SSL y frente a uno totalmente supervisado. El mejor
encoder se empaqueta como un modelo que da P(AD) para MRIs nuevas (`deploy/ad_mri_v2`).

## El pipeline en una página

```text
 1. datos       scripts/1_data/          MRI (LMDB, MNI 2 mm) → cubos 3D de los hipocampos
 2. splits      scripts/2_splits/        quién entrena SSL, quién tiene etiqueta, quién es test
 3. entrenar    train.py  (+ cloud/)     un encoder SSL por llamada (en GPU: RunPod)
 4. evaluar     src/t5_ssl/evaluation/   probe lineal AD vs HC sobre el encoder congelado
                scripts/4_evaluate/      re-evaluar y comparar corridas
 5. reportar    scripts/5_reports/       tablas y figuras de cada campaña
 6. desplegar   scripts/6_deploy/        el modelo final y P(AD) para MRIs nuevas
```

El recorrido completo, con qué archivo hace cada paso y qué produce:
**[`docs/PIPELINE.md`](docs/PIPELINE.md)**.

```bash
python train.py --model byol --dry-run      # la configuración que se entrenaría
python train.py --model byol --epochs 50 --seed 1 \
    --set data.manifest_path=campaign_ssl3d_runpod/split_deploy/manifest.csv
python scripts/6_deploy/predict_ad.py --model deploy/ad_mri_v2 --volumes scan.nii.gz
```

## Dónde está cada cosa

```text
oasis3-ssl-workbench/
├── train.py              entrada principal: entrena y evalúa un modelo SSL
├── src/t5_ssl/           el paquete
│   ├── recipes.py        recetas de las campañas (método, variante, vistas, ROI, encoder)
│   ├── train.py          construcción y entrenamiento (datos → encoder → método → Trainer)
│   ├── data/             datamodules, cubos ROI 3D, vistas/augmentations
│   ├── models/           encoders 2D/3D, cabezas, módulos VICReg/BYOL/DINOv2/pretexto
│   ├── losses/           pérdidas de cada método
│   ├── evaluation/       probes (v1 y v2), métricas, identidad, QC de entrada
│   ├── callbacks/        salud de la representación, colapso
│   ├── utils/            runtime rápido de GPU, métricas de representación, logging
│   └── cli.py            el comando `t5-ssl` (data-info, train)
├── configs/              configuración Hydra (ver configs/README.md)
├── scripts/              un script por tarea, en carpetas por etapa (ver scripts/README.md)
│   └── legacy/           scripts de campañas anteriores, archivados
├── cloud/                RunPod: crear el pod, instalar, preflight, colas de corridas
├── tests/                pruebas (pytest)
├── tools/                visualizadores y notebook del dataset
├── docs/                 guías, informes y documentos históricos (docs/historia/)
├── deploy/               modelos desplegables con su MODEL_CARD.md
└── campaign_*/           resultados de cada campaña (ver docs/CAMPANAS.md)
```

| Para saber... | Leer |
|---|---|
| qué hace cada paso y con qué código | [`docs/PIPELINE.md`](docs/PIPELINE.md) |
| dónde quedó cada archivo tras la reorganización del 2026-11-16 | [`docs/MAPA_DEL_CODIGO.md`](docs/MAPA_DEL_CODIGO.md) |
| qué es cada carpeta `campaign_*` y dónde está su conclusión | [`docs/CAMPANAS.md`](docs/CAMPANAS.md) |
| qué script usar en cada etapa | [`scripts/README.md`](scripts/README.md) |
| qué configuraciones Hydra usa el pipeline actual | [`configs/README.md`](configs/README.md) |
| el modelo desplegado, sus métricas y límites | `deploy/ad_mri_v2/MODEL_CARD.md` |
| la última conclusión (validación externa de 5 pliegues) | `campaign_ssl3d_runpod/outer5/REPORT.md` |
| cómo eran las campañas 2D de septiembre | [`docs/historia/README_campanas_2D.md`](docs/historia/README_campanas_2D.md) |

## Instalación

Python ≥ 3.10, en un entorno virtual:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[analysis,tuning,toolkit]" pytest ruff
.venv/bin/t5-ssl data-info          # encuentra el CSV y el LMDB
.venv/bin/python -m pytest          # pruebas
bash scripts/verify_local.sh        # lint + pruebas + una lectura del LMDB (no entrena)
```

En Windows el intérprete es `.venv/Scripts/python.exe`. Para PyTorch sólo CPU, instalar
antes `torch torchvision --index-url https://download.pytorch.org/whl/cpu`. En GPU
(RunPod, CUDA 12): `cloud/setup.sh` o el `Dockerfile`.

## Datos

Los datos derivados de OASIS3 **no se publican**: el CSV clínico, el LMDB, los caches de
cubos, los manifiestos con sujetos, los checkpoints y los resultados por sujeto quedan
fuera de Git (`.gitignore`). Por defecto se usa `data/dementia/mni-dataset.csv` y
`data/dementia/mni-dataset.lmdb`; las rutas se cambian con las variables de
[`.env.example`](.env.example). Contrato de datos: [`data/README.md`](data/README.md);
checkpoints externos: [`artifacts/README.md`](artifacts/README.md).
