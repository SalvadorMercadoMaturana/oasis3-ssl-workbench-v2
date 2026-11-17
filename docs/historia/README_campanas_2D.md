> **Documento histórico.** Este era el `README.md` de la raíz hasta la reorganización del
> 2026-11-16. Describe las campañas de septiembre de 2026 (cortes 2D del LMDB y
> `python -m t5_ssl.train`); el pipeline actual está en [`README.md`](../../README.md) y
> [`docs/PIPELINE.md`](../PIPELINE.md). Sólo se actualizaron las rutas: los scripts que
> menciona están ahora en `scripts/legacy/` (o en `cloud/`) y los enlaces apuntan desde esta
> carpeta. El resto del texto quedó como estaba.

# OASIS3 SSL Workbench

Repositorio cohesionado para inspeccionar OASIS3, leer el LMDB/NIfTI, reproducir evaluaciones y
desarrollar VICReg, BYOL y una variante compacta de DINOv2. El código entrenable vive en
`src/t5_ssl`; las utilidades educativas y el notebook están en `tools/dataset_oasis3`.

Los datos privados y los checkpoints **no se publican en Git**. Por defecto se detecta el checkout
local `data/dementia`; también puede configurarse con las variables de `.env.example`. Consulta
[`data/README.md`](../../data/README.md) para el contrato de datos y
[`artifacts/README.md`](../../artifacts/README.md) para checkpoints; el mapa completo está en
[`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md).

Este directorio contiene el código T5 del ZIP adaptado a OASIS3. VICReg, BYOL y DINOv2 compacto conservan sus backbones, proyectores, pérdidas y LightningModules originales. La variante DINOv2 incluye destilación de CLS, parches enmascarados y KoLeo; **no es una reproducción completa de DINOv2**. La campaña reproducible usa las imágenes MNI del LMDB original de t4.

La inspección local encontró 1.724 sesiones OASIS3, 2.567 archivos T1w y 981 sujetos en el CSV clínico. Un NIfTI examinado tenía forma `(129, 137, 161)`, tipo `uint8`, rango `0..255` y orientación LIA. No había un checkout Git accesible en `~/`; la adaptación vive sólo en este directorio extraído del ZIP.

El entorno observado usa Python 3.10 y CPU (el driver GPU no es compatible con el PyTorch instalado). Instalar en un entorno virtual:

```bash
cd ~/oasis3-ssl-workbench
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[analysis,tuning,toolkit]" pytest ruff
```

### Uso local y rutas portables

La configuración ya no depende de las rutas absolutas del servidor. Detecta por defecto
`data/dementia/mni-dataset.csv` y `data/dementia/mni-dataset.lmdb` dentro del workbench. Las rutas pueden sobrescribirse con las variables documentadas en
`.env.example` o mediante overrides de Hydra.

En WSL, desde la raíz del repositorio:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e ".[analysis,tuning,toolkit]" pytest ruff
.venv/bin/t5-ssl data-info
.venv/bin/python -m pytest
bash scripts/verify_local.sh
```

Los datasets, manifiestos con sujetos, logs y checkpoints se mantienen locales mediante `.gitignore` y
no deben añadirse a Git sin revisar primero sus permisos de redistribución.

El DataModule une el CSV configurado con los NIfTI de `OASIS3_IMAGE_ROOT` mediante `original_code`, filtra `study_name == OASIS3` y admite varias adquisiciones por sesión. No escribe en esos directorios. Reorienta cada NIfTI a RAS (los archivos observados son LIA, cuyo tercer eje no es axial), convierte cada volumen 3D a cortes axiales 2D, normaliza entre los percentiles 1.º y 99.º del tejido no nulo a `[0,1]` (también admite intensidades firmadas), y por defecto extrae cinco cortes entre el 35 y 65 % del eje axial. `data.channels=3` usa tres cortes vecinos, junto con `model.backbone.in_channels=3`. Los sujetos se reparten de forma determinista y estratificada (70/15/15); todas sus sesiones y adquisiciones quedan en el mismo split. El entrenamiento SSL recibe sólo imágenes y metadatos; `group` nunca entra en la pérdida.

Datos y prueba breve de lectura:

```bash
.venv/bin/t5-ssl data-info
OMP_NUM_THREADS=2 PYTHONPATH=src .venv/bin/python scripts/legacy/smoke_oasis3.py
```

El smoke ejecuta **un batch de entrenamiento y uno de validación por método** con dos volúmenes por split, comprueba vistas, ausencia de gradientes en profesores, splits y recarga de checkpoints. Sus pérdidas no son métricas de baseline.

Entrenamiento alternativo sobre los NIfTI originales, desde este directorio (sin `data.max_volumes` ni límites de batches):

```bash
OMP_NUM_THREADS=4 .venv/bin/t5-ssl train trainer.accelerator=cpu trainer.max_epochs=50
OMP_NUM_THREADS=4 .venv/bin/t5-ssl train model=byol_convnext loss=byol trainer.accelerator=cpu trainer.max_epochs=50
OMP_NUM_THREADS=4 .venv/bin/t5-ssl train model=dinov2_vit loss=dinov2 augmentation=dinov2_multicrop trainer.accelerator=cpu trainer.max_epochs=50
```

Cada ejecución crea `logs/<método>.../` con `config.yaml`, `subject_splits.csv`, métricas CSV y checkpoints `best-loss`/`last`; `last.ckpt` se guarda explícitamente al terminar `fit` y corresponde a la última época. En hardware CUDA compatible, se puede usar `trainer.accelerator=gpu` y ajustar batch, trabajadores y precisión. Para la opción de tres cortes vecinos, añadir `data.channels=3 model.backbone.in_channels=3` a cada comando.

Evaluación AD frente a HC, **después de terminar** cada entrenamiento:

```bash
PYTHONPATH=src .venv/bin/python scripts/legacy/evaluate_frozen_probe.py logs/<directorio-del-experimento>
```

El probe congela el encoder, agrega embeddings de los cortes y sesiones seleccionados por sujeto, excluye sujetos con etiquetas ambiguas o distintas de AD/HC, ajusta regresión logística con train, escoge `C` por AUROC de val y evalúa test una sola vez. Repite el protocolo con un backbone aleatorio de la misma arquitectura y semilla como control. Guarda `ad_hc_subject_probe.json`. Las pruebas smoke no constituyen un baseline completo ni producen métricas AD/HC fiables.

Rutas alternativas: `data.csv_path=/ruta/tabla.csv data.image_root=/ruta/imagenes`. El loader exige la estructura de columnas `study_name`, `original_code`, `group` y nombres T1w que empiecen por el código de sesión. Si cambian los datos entre entrenamiento y evaluación, el probe aborta cuando el split recalculado no coincide con `subject_splits.csv`.

## Cohort LMDB y herramientas de la campaña solicitada

El `scripts/loader.py` del dataset local fue inspeccionado: es idéntico byte a byte a `dataset_report.py` y no contiene un `Dataset` ni acceso al LMDB. Se conserva en el checkout privado. `data=oasis3_lmdb` activa un adaptador **nuevo** que lee el LMDB en modo de sólo lectura. Usa sus claves `idx` del CSV y comprueba que la sesión y el estudio coincidan. Es el loader de esta adaptación, no el dataloader Python original ausente.

`representative_oasis3.csv` selecciona 98 sujetos OASIS3 de forma determinista (68 train, 14 val, 16 test), estratificados por AD, HC y otros. Asigna todas las sesiones del mismo sujeto a un único split y marca `primary=1` para la primera sesión. El modo LMDB de campaña usa una sesión por sujeto, un corte axial aleatorio entre el 30 y 70 % del volumen en cada visita de entrenamiento y el corte central fijo para validación y probe. `representative_data_check.json` confirma que las 98 claves existen en el LMDB y que sus imágenes son `int16` de `(97,115,97)`. La inspección visual está en `lmdb_axial_qc.png`.

```bash
.venv/bin/python scripts/legacy/select_representative_subjects.py --fraction 0.10 --seed 42
.venv/bin/python scripts/legacy/extend_heldout_test.py
bash scripts/legacy/run_100_epochs.sh
OMP_NUM_THREADS=2 .venv/bin/python scripts/legacy/finalize_campaign.py
```

La campaña ejecuta 100 épocas completas por método en CPU. `expanded_test_oasis3.csv` conserva los 68 sujetos de train y los 14 de val, y añade al test los demás sujetos OASIS3 no vistos por el entrenamiento (899 sujetos de test en total, incluida la pequeña partición original); quedó fijado antes de calcular cualquier resultado de test. De ellos, el probe admite 806 con etiqueta AD/HC inequívoca (186 AD, 620 HC); train admite 61 y val 13. `finalize_campaign.py` verifica que estén las 100 épocas de train y val, calcula el probe AD/HC una sola vez por método y escribe `campaign_100ep/results.json`, `campaign_100ep/RESULTADOS.md` y los gráficos en `campaign_100ep/figures/`. La comparación usa un encoder con pesos aleatorios de la misma arquitectura, con el mismo split y protocolo de probe. Es un piloto con preentrenamiento sobre 68 sujetos, no un baseline SSL del dataset completo. Para preentrenar con todos los sujetos y sesiones OASIS3, pasar `data.manifest_path=null data.primary_only=false`, aceptar mayor tiempo y almacenar el split guardado con cada experimento. El test ampliado aquí definido no debe reutilizarse en esa campaña porque incluiría sujetos de entrenamiento.

La primera evaluación de esta campaña se descartó porque el probe tomaba cortes de train aleatorios. El evaluador se corrigió para reconstruir train, val y test con el corte central fijo, y se repitió la evaluación sin cambiar checkpoints ni hiperparámetros. Los archivos de resultados publicados corresponden sólo al probe corregido. `run_100_epochs.sh` omite métodos ya completados si se vuelve a ejecutar.

Resultados medidos: [tabla y protocolo](../../campaign_100ep/RESULTADOS.md), [curvas SSL](../../campaign_100ep/figures/ssl_learning_curves.png), [AUROC en test](../../campaign_100ep/figures/ad_hc_subject_auc.png) y [diferencia pareada frente al control](../../campaign_100ep/figures/ad_hc_delta_auc.png). En esta única partición y semilla, DINOv2 compacto mejoró +0,034 AUROC frente a su backbone aleatorio; los intervalos de VICReg y BYOL incluyen cero. El preentrenamiento SSL usó sólo 68 sujetos, así que esto sigue siendo un piloto, no una conclusión general para todo OASIS3.

`scripts/legacy/plot_experiments.py` genera las curvas de train/val y el gráfico de AUROC entrenado frente al control aleatorio. `scripts/legacy/tune_ssl.py` ofrece búsqueda Optuna reanudable por método y cohorte, con SQLite; permite maximizar AUROC de sujetos en val o minimizar pérdida SSL de val sin usar test. Ejemplo:

```bash
.venv/bin/python scripts/legacy/tune_ssl.py --method vicreg --objective val_auc --trials 20 --epochs 15 data=oasis3_lmdb trainer.accelerator=cpu
.venv/bin/python scripts/legacy/tune_ssl.py --method byol --objective val_auc --trials 20 --epochs 15 data=oasis3_lmdb trainer.accelerator=cpu
.venv/bin/python scripts/legacy/tune_ssl.py --method dinov2 --objective val_auc --trials 20 --epochs 15 data=oasis3_lmdb trainer.accelerator=cpu
.venv/bin/python scripts/legacy/plot_experiments.py logs/<vicreg> logs/<byol> logs/<dinov2> --output-dir figures_100ep
```

`figures_smoke/`, `tuning_smoke/` y `tuning_smoke_lmdb/` sólo verifican que los scripts funcionan; sus gráficos y métricas proceden de pruebas de un paso, no de una campaña de 100 épocas.

## Tuning de la tarea de pretexto sin probe

`scripts/legacy/tune_pretext.py` es la búsqueda para mejorar **sólo** el objetivo SSL. Fija las pérdidas nativas de VICReg, BYOL y DINOv2 compacto, de modo que `val/loss` tenga la misma definición dentro de cada estudio. Optuna TPE prueba tasa de aprendizaje, weight decay, recorte mínimo y ruido de augmentación; incluye la configuración original como ensayo 0, observa la media de las últimas cinco pérdidas de validación y poda ensayos malos después de 20 épocas. Cada estudio usa el mismo manifiesto OASIS3 por sujeto; las etiquetas sólo se usaron para estratificar ese split previo, nunca para seleccionar hiperparámetros. No carga el probe ni consulta test. SQLite permite reanudar la búsqueda.

```bash
OMP_NUM_THREADS=2 .venv/bin/python scripts/legacy/tune_pretext.py --method vicreg --trials 12 --epochs 50 --min-epochs 20 data=oasis3_lmdb trainer.accelerator=cpu
OMP_NUM_THREADS=2 .venv/bin/python scripts/legacy/tune_pretext.py --method byol --trials 12 --epochs 50 --min-epochs 20 data=oasis3_lmdb trainer.accelerator=cpu
OMP_NUM_THREADS=2 .venv/bin/python scripts/legacy/tune_pretext.py --method dinov2 --trials 12 --epochs 50 --min-epochs 20 data=oasis3_lmdb trainer.accelerator=cpu
OMP_NUM_THREADS=2 .venv/bin/python scripts/legacy/refit_pretext.py --method vicreg
OMP_NUM_THREADS=2 .venv/bin/python scripts/legacy/refit_pretext.py --method byol
OMP_NUM_THREADS=2 .venv/bin/python scripts/legacy/refit_pretext.py --method dinov2
.venv/bin/python scripts/legacy/report_pretext_tuning.py
```

Las pérdidas SSL de métodos diferentes no se comparan entre sí. Una pérdida menor no garantiza un mejor probe AD/HC; este estudio sólo mide el ajuste de la tarea de pretexto en la validación definida.

La campaña guardada en `tuning_pretext/` se detuvo a petición del usuario tras nueve ensayos resueltos para VICReg y BYOL; DINOv2 completó doce. Para reproducir los tres modelos seleccionados a 100 épocas, ejecute únicamente los comandos `refit_pretext.py` y `report_pretext_tuning.py` del bloque anterior. Los comandos `tune_pretext.py` corresponden a una búsqueda nueva o a continuar una existente.

## Campaña completa OASIS3: 50 épocas por método

`data=oasis3_full50` usa el CSV y LMDB originales de t4. El manifiesto fijado con semilla 42 incluye los 981 sujetos OASIS3: 685 train, 146 val y 150 test, sin solapamiento. Las 1.226 sesiones de train están disponibles para SSL; en cada época se toma una sesión aleatoria y un corte axial aleatorio por sujeto. Así se aprovechan sesiones longitudinales sin multiplicar por 1.226 los pasos de cada época. Val y test usan la sesión primaria y el corte central fijos. El grupo clínico se usa únicamente para estratificar el split y después para el probe AD/HC, nunca en la pérdida SSL. El cache de volúmenes normalizados evita releer el LMDB en cada época.

Se mantienen las configuraciones nativas de modelo, augmentación y optimizador de `configs/`; esta campaña **no** usa los hiperparámetros Optuna del piloto pequeño. Para reproducirla desde el directorio del proyecto:

```bash
.venv/bin/python scripts/legacy/select_representative_subjects.py --fraction 1.0 --seed 42 --output campaign_full50/all_sessions.csv
.venv/bin/python - <<'PY'
import csv
with open('campaign_full50/all_sessions.csv', newline='') as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    rows = [r for r in reader if r['split'] == 'train' or r['primary'] == '1']
with open('campaign_full50/train_sessions_eval_primary.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
PY
bash scripts/legacy/run_full50.sh
OMP_NUM_THREADS=2 .venv/bin/python scripts/legacy/finalize_full50.py
```

`run_full50.sh` lanza VICReg, BYOL y DINOv2 compacto en paralelo, con máximo 50 épocas completas. Tras la indicación de aplicar early stopping, `trainer=full50` detiene cada corrida tras diez épocas sin reducir `val/loss` SSL al menos 0,001; no mira el probe. Los experimentos que ya estaban en ejecución al recibir esa indicación se vigilan con la misma regla y quedan registrados en `campaign_full50/early_stopping.json`. En un servidor con GPU operativa puede reemplazarse `trainer.accelerator=cpu` por `gpu` en ese script. Los checkpoints `last.ckpt` y `best-loss` quedan en `logs/`; `finalize_full50.py` verifica las épocas realmente ejecutadas, usa el checkpoint final para corridas completas y el mejor SSL para las detenidas, ajusta un probe lineal con encoder congelado sobre sujetos AD/HC de train, elige su regularización en val y evalúa test una vez. Contrasta con la misma arquitectura aleatoria y con el prior AD sin imagen. Escribe [resultados](../../campaign_full50/RESULTADOS.md), gráficos de pérdida SSL, AUROC y log-loss en `campaign_full50/figures/`, más JSON con predicciones por sujeto. No use el `expanded_test_oasis3.csv` del piloto anterior: contiene sujetos de train de esta campaña.

## Campaña MRI con campo completo: entrenamiento interrumpido

`mri_fullview20` conserva todo el corte axial para VICReg y BYOL; `mri_fullview20_dino` hace lo mismo para DINOv2 y desactiva sus crops locales. Ambas políticas eliminan flips y blur, y aplican sólo rotación ±5°, intensidad ×[0,95, 1,05] y ruido σ=0,01. Las dos vistas de validación son deterministas y usan una rotación suave en lugar de un flip que no aparece en entrenamiento. Se usa el mismo manifiesto OASIS3 de `campaign_full50` y los mismos modelos y optimizador.

```bash
bash scripts/legacy/run_mri_fullview20.sh
OMP_NUM_THREADS=2 .venv/bin/python scripts/legacy/finalize_mri_fullview20.py
```

La campaña se lanzó con máximo 20 épocas. DINOv2 completó 20 antes de que se pidiera detener los restantes a 15; VICReg completó 15 y BYOL se detuvo después de 12 por una segunda petición. El intento de reanudar BYOL no terminó ninguna época adicional y no contribuye a sus resultados. El [manifiesto de ejecución](../../campaign_mri_fullview20/experiment_manifest.json) fija los directorios y las épocas efectivas; el script de lanzamiento reconoce las corridas detenidas y no las reinicia al repetirse. El finalizador verifica las épocas efectivas y la identidad de sujetos y registros frente a `campaign_full50`, selecciona el checkpoint por `val/loss` SSL, evalúa una vez el test AD/HC con un probe congelado y genera las mismas cinco figuras de la campaña anterior, más una comparación visual de augmentations. Los resultados quedan en `campaign_mri_fullview20/`. Una diferencia respecto a la campaña de 50 épocas no aísla el efecto de la augmentation porque también cambia la duración del entrenamiento.

## Entrenamiento rápido en GPU (RTX 4090 / RunPod)

Entrypoints opcionales que no cambian las configs ni `t5-ssl-train`:

| Archivo | Para qué |
|---|---|
| `train.py --model byol\|vicreg\|dinov2 [--variant ...]` | Un modelo, protocolo de `campaign_ssl3d_v2` (split, vistas, probes) con el runtime rápido: bf16, TF32, AdamW fused, `channels_last_3d`, `torch.compile` con fallback, workers persistentes; reintento ante OOM con batch/2 y acumulación ×2. |
| `cloud/run_all.py --parallel mps\|all-at-once\|sequential` | Varios trabajos en una GPU (procesos `spawn`, MPS si está disponible), logs por trabajo, `REPORT.md` con img/s, s/época, VRAM pico y probes. `--preset byol-screen` = experimentos de BYOL. |
| `cloud/bench.py` | Throughput real de 100 pasos por modelo y config (baseline vs rápido vs compile). |
| `cloud/preflight.py` | GPU, VRAM, versiones, datos, compile. |
| `cloud/setup.sh`, `cloud/bootstrap_runpod.sh`, `cloud/env.sh`, `Dockerfile` | Instalación sobre CUDA 12 + PyTorch ≥ 2.4 y punto de entrada único en el pod. |
| `cloud/runpod_pod.py` | Crea (≤ `--max-price` USD/h), consulta, mide el costo y termina el pod. |

El perfil de runtime es `configs/perf/rtx4090.yaml` (`+perf=rtx4090`); sin GPU todo degrada a fp32/eager. Variantes de BYOL (`t5_ssl/recipes.py`): `border` (rotación con relleno de borde), `border_wide`, `border_gn`, `border_predlr` y `border_h3` (las tres). Salidas en `runs/` (gitignorado).
