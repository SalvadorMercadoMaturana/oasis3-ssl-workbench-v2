# Arquitectura

```text
oasis3-ssl-workbench/
├── train.py                  entrena y evalúa un modelo SSL (entrada de las campañas 3D)
├── src/t5_ssl/               paquete instalable y CLI (`t5-ssl`)
│   ├── recipes.py            método / variante / vistas / ROI / encoder → overrides de Hydra
│   ├── train.py              datos → encoder → módulo SSL → Trainer (también `python -m t5_ssl.train`)
│   ├── data/                 datamodules (LMDB 2D, NIfTI, cubos ROI 3D), vistas, rutas
│   ├── models/               encoders, cabezas y LightningModules de cada método
│   ├── losses/               VICReg, BYOL, DINOv2, MAE (el pretexto la calcula en su módulo)
│   ├── evaluation/           probe v1 y v2, métricas, embeddings, identidad, QC de entrada
│   ├── callbacks/            salud de la representación y monitor de colapso
│   └── utils/                runtime de GPU, métricas de representación, logging
├── configs/                  configuración Hydra (configs/README.md)
├── scripts/<etapa>/          1_data, 2_splits, 3_train, 4_evaluate, 5_reports, 6_deploy
├── scripts/legacy/           scripts de campañas anteriores (archivados, siguen corriendo)
├── cloud/                    RunPod: pod, instalación, preflight, colas de corridas, bench
├── tests/                    pruebas unitarias y de integración sintéticas
├── tools/dataset_oasis3/     visualizadores, estadísticas y notebook
├── artifacts/                contrato para checkpoints externos (sin binarios en Git)
├── data/                     datos privados locales (ignorados por Git)
├── deploy/                   modelos desplegables (pesos ignorados por Git)
└── campaign_*/               informes y figuras de cada campaña (docs/CAMPANAS.md)
```

Dependencias entre capas: los scripts de una etapa importan el paquete `t5_ssl` y, como
mucho, a sus vecinos de la misma carpeta; nunca a otra etapa (lo comprueba
`tests/test_scripts_layout.py`). Lo que comparten varias etapas vive en el paquete.

Flujo de datos: el CSV identifica sesiones y sujetos; el DataModule resuelve cada fila contra
el LMDB (o un cache de cubos ROI) y los splits se hacen por sujeto para evitar fuga
longitudinal. El modelo SSL recibe imágenes y metadatos, nunca la etiqueta clínica; los
evaluadores congelan el encoder y usan las etiquetas únicamente en el probe posterior.

`scripts/verify_local.sh` comprueba formato, tests, descubrimiento del dataset y una lectura
LMDB. No inicia `Trainer.fit`, Optuna ni ninguna campaña.

El checkout local dispone de `.venv` para modelos/pruebas y `.venv-toolkit` para Jupyter. Ambos
están dentro del workbench e ignorados por Git; una clonación nueva puede crear un solo `.venv`
instalando los extras `analysis,tuning,toolkit` descritos en el [`README`](../README.md).

Recorrido completo: [`PIPELINE.md`](PIPELINE.md). Dónde quedó cada archivo:
[`MAPA_DEL_CODIGO.md`](MAPA_DEL_CODIGO.md).
