# El pipeline de punta a punta

Qué hace el workbench, en el orden en que ocurre, con el archivo que hace cada paso.
El objetivo científico: **¿aprende un encoder autosupervisado (SSL), a partir de la
resonancia magnética sola, algo que distinga Alzheimer (AD) de sujetos sanos (HC)?**
Se responde entrenando el encoder sin etiquetas y midiendo después, con el encoder
congelado, cuánto rinde un clasificador lineal (el *probe*) sobre sus embeddings.

```text
 OASIS3 (CSV + LMDB, MNI 2 mm)
        │
        ▼  1. datos           scripts/1_data/        → data/roi3d/<cache>.{npy,json}
 cubos 3D del hipocampo (par izquierdo / derecho espejado)
        │
        ▼  2. splits          scripts/2_splits/      → campaign_ssl3d_runpod/split_*/manifest.csv
 qué sujetos entrenan SSL, cuáles tienen etiqueta, cuáles son test
        │
        ▼  3. entrenamiento   train.py  (+ cloud/ en GPU, scripts/3_train/)
 encoder SSL (VICReg, BYOL, DINOv2, pretexto) ─┐
        │                                      │ dentro de train.py
        ▼  4. evaluación      src/t5_ssl/evaluation/  → result.json (+ embeddings.npz, encoder.pt)
 probe lineal sobre el encoder congelado; scripts/4_evaluate/ re-evalúa y compara
        │
        ▼  5. reportes        scripts/5_reports/     → <campaña>/REPORT.md, figures/, docs/
        │
        ▼  6. despliegue      scripts/6_deploy/      → deploy/ad_mri_v2/ y P(AD) para MRIs nuevas
```

Todo lo que dice "campaña" en este documento se resume en [`CAMPANAS.md`](CAMPANAS.md);
dónde quedó cada archivo antes y después de la reorganización del 2026-11-16 está en
[`MAPA_DEL_CODIGO.md`](MAPA_DEL_CODIGO.md).

---

## 0. Los datos de partida

| Archivo | Qué es |
|---|---|
| `data/dementia/mni-dataset.csv` | Una fila por sesión: `idx` (clave del LMDB), sujeto, estudio (OASIS3, ADNI, ...), diagnóstico (`group`), edad, sexo. |
| `data/dementia/mni-dataset.lmdb` | Los volúmenes T1 registrados a MNI 2 mm, forma `(97, 115, 97)`. |

Son datos derivados de OASIS3: nunca entran a Git (ver `.gitignore`). Las rutas por
defecto están en `src/t5_ssl/data/paths.py` y se pueden cambiar con variables de entorno
(`.env.example`).

## 1. Datos: cubos de la región de interés (ROI)

Las campañas actuales no usan el cerebro entero sino **dos cubos alrededor de los
hipocampos** (32³ vóxeles evaluados; el cache guarda `round(1.2·32) = 38³` para que las
vistas puedan recortar y rotar). El cubo derecho se espeja, así ambos canales tienen la
misma orientación anatómica.

| Paso | Código |
|---|---|
| Construir un cache (`hippo_s32`, `multi3_s32`, `atrophy5_s32`, ...) | `scripts/1_data/build_roi_caches.py` → `t5_ssl.data.roi3d.build_cache` / `build_region_cache` |
| Añadir sesiones a un cache existente (mismas posiciones de cubo, filas viejas intactas) o crear el de otra cohorte (`--into adni_`) | `scripts/1_data/extend_roi_caches.py` → `t5_ssl.data.roi3d.crop_cubes` |
| Auditar etiquetas y desbalance demográfico de OASIS3 | `scripts/1_data/audit_oasis3_labels.py` |

Salida: `data/roi3d/<cache>.npy` (cubos float16, una fila por sesión) y
`data/roi3d/<cache>.json` (`idx` de cada fila y `cube_starts`, las esquinas de cada cubo).

## 2. Splits: quién entrena, quién tiene etiqueta, quién es test

Un split es un `manifest.csv` con una fila por sesión utilizable: `subject`, `idx`,
`split` (`train` / `val` / `test` / `test_<nombre>`), `labeled` (1 = el probe puede usar su
etiqueta AD/HC; 0 = sólo SSL) y `primary` (la sesión que se evalúa). Las etiquetas
**nunca** entran a la pérdida SSL: sólo al probe.

| Script | Split | Para qué |
|---|---|---|
| `scripts/2_splits/make_split_361_63_84.py` | `split_361_63_84` | 361 train / 63 val / 84 test, estratificado por diagnóstico y edad (campañas de octubre). |
| `scripts/2_splits/make_split_imbalanced.py` | `split_361_imb`, `split_agematch` | Test 1:6 AD:HC y split emparejado por edad 55/45. |
| `scripts/2_splits/make_split_deploy.py` | **`split_deploy`** (+ `external_adni.csv`) | El split actual: test representativo (41 AD + 137 HC, prevalencia y edades reales), train/val etiquetados emparejados por edad 1:1, 509 sujetos sólo-SSL; ADNI como test externo. |
| `scripts/2_splits/make_split_outer.py` | `split_outer5/fold0..4` | Validación cruzada externa: cada sujeto AD/HC es test una vez (fold 0 = `split_deploy`). |

## 3. Entrenamiento SSL: `train.py`

Una llamada entrena **y evalúa** un modelo:

```bash
python train.py --model byol --variant wide_dproj0.1 --aug-recipe noise \
    --backbone convnext_tiny --roi 32 --epochs 50 --seed 1 \
    --set data.manifest_path=campaign_ssl3d_runpod/split_deploy/manifest.csv \
    --save-embeddings --export-encoder --out-root runs --tag mi_prueba
python train.py --model byol --dry-run      # sólo muestra la configuración compuesta
```

Lo que pasa adentro, paso a paso:

1. **Receta → configuración.** `train.py:build_overrides` le pasa a
   `t5_ssl.recipes.overrides` el método (`--model`: `byol`, `vicreg`, `dinov2`, `pretext`),
   la variante (`--variant`, `recipes.VARIANTS`), la receta de vistas (`--aug-recipe`,
   `recipes.AUG_RECIPES`), el encoder (`--backbone`, `recipes.BACKBONES`), la ROI (`--roi`,
   `recipes.ROIS`) y el calendario; el resultado son *overrides* de Hydra sobre
   `configs/train.yaml` (ver [`configs/README.md`](../configs/README.md)). `train.py:compose`
   arma la configuración.
2. **Entrenamiento.** `t5_ssl.train.train(cfg)` (`src/t5_ssl/train.py`), en este orden:
   carpeta del experimento → vistas (`t5_ssl.data.roi3d.build_roi3d_transforms`) →
   datamodule (`ROI3DDataModule`) → encoder (`build_backbone`, en
   `t5_ssl.models.backbones.factory`: ResNet-18, ConvNeXt-T o ViT-S en 3D con pesos
   ImageNet/DeiT inflados, y la forma de pasarle la ROI; ver
   [Encoders e ingesta](#encoders-e-ingesta)) → módulo del método
   (`build_ssl_module`: `t5_ssl.models.*_module` + `t5_ssl.losses.*`) → callbacks →
   `Trainer.fit`. Si la GPU se queda sin memoria, `train.py:run` reintenta con la mitad del
   batch y el doble de acumulación (mismo batch efectivo).
3. **Evaluación del encoder congelado** (sección 4), salvo `--no-eval`.
4. **Resultados.** `<out-root>/<tag>/<nombre de corrida>/` con `result.json`,
   `metrics.csv` (curvas), `config.yaml`, `subject_splits.csv`, `data_records.csv`, y según
   las opciones `embeddings.npz`, `encoder.pt`, `collapse.json`.

Controles: `--init-only` evalúa el encoder **antes** del SSL (el "sin SSL" de las figuras:
los mismos pesos ImageNet/DeiT, congelados). El baseline totalmente supervisado es
`scripts/3_train/train_supervised.py` (el mismo encoder entrenado de punta a punta con las
etiquetas, por folds).

### Encoders e ingesta

`--backbone <nombre>` elige una entrada de `recipes.BACKBONES`: overrides de
`model.backbone` que `build_backbone` convierte en el módulo. Hay tres familias:

| Familia | Nombres | Qué cambia | Código |
|---|---|---|---|
| Encoder de cada método | `convnext_tiny`, `resnet18`, `vit_small` | — (lo de siempre) | `resnet3d.py`, `vit3d.py`, `inflate3d.py` |
| Otros pesos de partida | `vit_small_dv2` (DINOv2), `convnext_tiny_in22k`, `convnext_tiny_in22k_ft1k` (ImageNet-22k, 2026-11-23) | sólo los pesos 2D que se inflan | `vit3d.py`, `convnext_weights.py` |
| Ingesta (2026-11-22) | `cnx3d_*`, `rn18_3d_*`, `mednet*`, `cnx2d_*`, `rn18_2d_*`, `rn50_2d_*` | normalización, hemisferios por separado, remuestreo, cortes 2D, MedicalNet | `ingest.py`, `medicalnet.py` |
| Cerebro completo en 3 planos | `tri2d_*` (sólo con `--roi brain2d`) | axial, coronal y sagital por el centro de masa | `triplane.py`, `t5_ssl/data/brain2d.py` |

Todo el código está en `src/t5_ssl/models/backbones/`. Líneas de cola de estas
comparaciones: `scripts/3_train/jobs_ingest.py`; informes: `scripts/5_reports/report_ingest.py`
(→ `campaign_ssl3d_runpod/ingest/REPORT.md`) y `report_brain2d.py` (→ `brain2d/REPORT.md`).
El multiplano de cerebro completo quedó descartado (ver `docs/CAMPANAS.md`).

### En GPU (RunPod)

| Archivo | Para qué |
|---|---|
| `cloud/runpod_pod.py` | Crear (con precio máximo), esperar, consultar costo y terminar el pod. |
| `cloud/setup.sh`, `cloud/bootstrap_runpod.sh`, `cloud/env.sh`, `Dockerfile` | Instalación en el pod y punto de entrada único (setup + preflight + lanzamiento). |
| `cloud/preflight.py` | GPU, VRAM, versiones, datos y compile antes de gastar. |
| `cloud/run_all.py` | Varias corridas de `train.py` en una GPU (presets de campaña, `REPORT.md`). |
| `cloud/queue_runner.py` + `scripts/3_train/jobs_outer5.py` | Cola de trabajos en paralelo (la campaña outer5 del 2026-11-16). |
| `cloud/bench.py`, `scripts/3_train/bench_augment_cost.py` | Throughput de GPU y costo de CPU de las vistas. |

## 4. Evaluación: el probe lineal (`src/t5_ssl/evaluation/`)

Siempre igual: el encoder congelado embebe el cubo central limpio de cada sesión, se
promedia por sujeto (`embeddings.py`) y una regresión logística estandarizada y
balanceada por clase predice AD vs HC. Hay dos protocolos, ambos dentro de `result.json`:

| Protocolo | Módulo | Qué mide |
|---|---|---|
| **v1** (campañas hasta 2026-11) | `probe_v1.py` | `cv`: CV estratificada 5×5 sobre los sujetos AD/HC de train+val; `test`: probe ajustado en train, `C` elegido en val, test una sola vez. |
| **v2** (desde 2026-11-07; el que decide) | `protocol_v2.py` | CV **representativa**: los folds etiquetados también puntúan a los sujetos AD/HC sólo-SSL de train (prevalencia y edades reales, como el test); `C` por log-loss interna; umbrales F1 fuera de fold; `auc_age` (AUROC sin el atajo de la edad); `mean3 = (AUROC + AUPR + F1-macro)/3` para seleccionar modelos. |

Otros módulos: `metrics.py` (AUROC, AUPR, F1, mean3, métricas a un umbral),
`identity.py` (`--identity`: ¿el encoder distingue sujetos bajo sus propias
augmentations o colapsó?), `run_scoring.py` (lo que `train.py` ejecuta tras entrenar),
`curves.py` (curvas por época), `input_qc.py` (rango de intensidad aceptado).

Después del entrenamiento:

| Script | Para qué |
|---|---|
| `scripts/4_evaluate/reeval_deploy.py` | Re-puntuar con el protocolo v2 corridas viejas de `split_deploy` (escribe `protocol_v2.json`). |
| `scripts/4_evaluate/compare_runs.py` | Diferencias entre grupos de corridas con bootstrap pareado. |
| `scripts/4_evaluate/analyze_cv_vs_test.py` | Por qué el test salía mejor que la CV (respuesta: azar de un único test de 41 AD). |
| `scripts/4_evaluate/tune_probe.py` | Variantes del probe (PCA, L1, elastic net) sobre la CV representativa. |
| `scripts/4_evaluate/trivial_baselines.py` | Líneas base sin aprendizaje de representación con el mismo protocolo v2. El volumen hipocampal solo ya iguala a los SSL; ver `docs/informe_senal_compartida.pdf`. |

## 5. Reportes

Cada campaña tiene su script en `scripts/5_reports/` que lee los `result.json` /
`protocol_v2.json` y escribe tablas (`REPORT.md`, `RESULTADOS.md`) y figuras. Los informes
PDF de `docs/` se arman con `build_report_*.py`; las figuras de la presentación final
(`docs/figures_final/`) con `present_final.py`. La lista completa por campaña está en
[`CAMPANAS.md`](CAMPANAS.md) y [`scripts/README.md`](../scripts/README.md).

## 6. Despliegue: el modelo que predice P(AD)

```bash
python scripts/4_evaluate/reeval_deploy.py --tags dep50_init dep50_ssl v2_ssl
python scripts/6_deploy/build_deploy_model.py --runs campaign_ssl3d_runpod/deploy/runs --out deploy/ad_mri_v2
python scripts/6_deploy/predict_ad.py --model deploy/ad_mri_v2 --volumes scan1.nii.gz scan2.npy
```

`build_deploy_model.py` elige, entre los encoders de las campañas de despliegue, el
ensamble de semillas con mayor `mean3` en la CV representativa (antes de mirar el test),
ajusta el probe final por semilla, fija umbrales y escribe el modelo con su
`MODEL_CARD.md` y `validation.json`. `predict_ad.py` aplica el modelo a MRIs nuevas en la
grilla MNI 2 mm: corta los cubos en las posiciones fijas del modelo, rechaza escaneos
fuera del rango de intensidad de OASIS3 (`t5_ssl.evaluation.input_qc`) y marca los fuera
de distribución. El modelo actual es `deploy/ad_mri_v2` (BYOL ConvNeXt-T con cabezas BN).

---

## Ruta antigua (2D, antes de octubre de 2026)

Las primeras campañas entrenaban sobre cortes 2D del LMDB o de los NIfTI con
`python -m t5_ssl.train` (Hydra directo, `configs/data/oasis3*.yaml`) y se evaluaban con
`scripts/legacy/evaluate_frozen_probe.py`. Ese camino sigue funcionando (el datamodule 2D,
los callbacks de salud de la representación), pero sus scripts de campaña están
archivados en `scripts/legacy/` (ver su [`README.md`](../scripts/legacy/README.md)) y su
documentación original en [`historia/README_campanas_2D.md`](historia/README_campanas_2D.md).
