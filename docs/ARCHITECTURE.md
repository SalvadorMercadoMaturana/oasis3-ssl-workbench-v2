# Arquitectura

```text
oasis3-ssl-workbench/
├── configs/                  configuración Hydra de datos, modelos y entrenamiento
├── artifacts/                contrato para checkpoints externos (sin binarios en Git)
├── data/dementia/            checkout privado local (ignorado por Git)
├── src/t5_ssl/               paquete instalable y CLI
├── scripts/                  evaluación, auditoría y reproducción de campañas
├── tests/                    pruebas unitarias y de integración sintéticas
├── tools/dataset_oasis3/     visualizadores, estadísticas y notebook
└── campaign_*/               informes y figuras publicables, sin checkpoints/logs
```

Flujo de datos: el CSV identifica sesiones y sujetos; el DataModule resuelve cada fila contra LMDB
o NIfTI; los splits se hacen por sujeto para evitar fuga longitudinal; el modelo recibe imágenes y
metadatos, nunca la etiqueta clínica en la pérdida SSL. Los evaluadores cargan un encoder congelado
y usan las etiquetas únicamente en el probe posterior.

`scripts/verify_local.sh` comprueba formato, tests, descubrimiento del dataset y una lectura LMDB.
No inicia `Trainer.fit`, Optuna ni ninguna campaña.

El checkout local dispone de `.venv` para modelos/pruebas y `.venv-toolkit` para Jupyter. Ambos
están dentro del workbench e ignorados por Git; una clonación nueva puede crear un solo `.venv`
instalando los extras `analysis,tuning,toolkit` descritos en el README.
