# Scripts, por etapa del pipeline

Se ejecutan desde la raíz del repositorio (`python scripts/<etapa>/<script>.py ...`); cada
uno trae su uso en el docstring de arriba (`python scripts/<etapa>/<script>.py --help`
cuando tiene argumentos). El recorrido completo está en
[`docs/PIPELINE.md`](../docs/PIPELINE.md). Los scripts de una carpeta sólo importan a sus
vecinos de la misma carpeta o al paquete `t5_ssl` (lo comprueba
`tests/test_scripts_layout.py`).

## `1_data/`: cubos de la región de interés

| Script | Qué hace |
|---|---|
| `build_roi_caches.py` | Construye caches de cubos (`hippo_s64`, `multi3_s32`, `atrophy5_s32`, ...) en `data/roi3d/`. |
| `extend_roi_caches.py` | Añade sesiones a un cache con sus mismas posiciones de cubo, o crea el de otra cohorte (`--into adni_`). |
| `audit_oasis3_labels.py` | Disponibilidad de etiquetas y desbalance demográfico de OASIS3. |

## `2_splits/`: manifiestos de split (`campaign_ssl3d_runpod/split_*/`)

| Script | Qué hace |
|---|---|
| `make_split_361_63_84.py` | 361 / 63 / 84 sujetos estratificados por diagnóstico y edad. |
| `make_split_imbalanced.py` | Test 1:6 AD:HC y split emparejado por edad 55/45 (sobre el 361). |
| `make_split_deploy.py` | **Split actual**: test representativo, train/val emparejados por edad, sólo-SSL, ADNI externo. |
| `make_split_outer.py` | Validación cruzada externa de 5 pliegues (fold 0 = `split_deploy`). |

## `3_train/`: lo que acompaña a `train.py`

| Script | Qué hace |
|---|---|
| `train_supervised.py` | Baseline totalmente supervisado: el mismo encoder entrenado de punta a punta con las etiquetas. |
| `jobs_outer5.py` | Genera las líneas de trabajo de la campaña outer5 para `cloud/queue_runner.py`. |
| `bench_augment_cost.py` | Costo de CPU de las vistas de entrenamiento por receta y ROI. |

El entrenamiento SSL en sí es `train.py` (raíz); en GPU, `cloud/`.

## `4_evaluate/`: re-evaluar y comparar corridas ya entrenadas

| Script | Qué hace |
|---|---|
| `reeval_deploy.py` | Re-puntúa corridas de `split_deploy` con el protocolo v2 (`protocol_v2.json`). |
| `compare_runs.py` | Comparaciones entre grupos de corridas con bootstrap pareado. |
| `analyze_cv_vs_test.py` | Por qué el test superaba a la CV en `split_deploy`. |
| `tune_probe.py` | Variantes del probe lineal sobre la CV representativa (sólo CPU). |
| `trivial_baselines.py` | Líneas base sin aprendizaje de representación (estadísticas de intensidad del cubo, volúmenes FreeSurfer, ConvNeXt-T aleatorio, ImageNet recalculado) con el protocolo v2 en los 5 pliegues externos. |

La evaluación de cada corrida ocurre dentro de `train.py` (`src/t5_ssl/evaluation/`).

## `5_reports/`: tablas, figuras e informes

| Script | Campaña (en `campaign_ssl3d_runpod/`) |
|---|---|
| `plot_ssl3d_runpod.py` | figuras generales de la campaña (lo importan los demás `plot_*`) |
| `report_aug_recipes3d.py`, `render_md_tables.py` | `augs20/`, `dropout20/` |
| `report_roi3d.py` | `roi50/` |
| `plot_champions.py` | `champions50/`, `split361_30/` |
| `plot_byol_diagnosis.py`, `report_byol_tuning.py`, `plot_byol_gn.py`, `plot_byol_pair.py` | `byol_diag30/`, `byol_tune30/`, `byol_fbn30/` |
| `build_report_split361.py`, `build_report_metodologia.py`, `plot_architectures_split361.py`, `plot_split_balance.py`, `analyze_latent_split361.py`, `plot_latent_split361.py` | split 361 → `docs/informe_split361.pdf`, `docs/informe_metodologia_split361.pdf` |
| `report_split361_100.py` | `split361_100/` |
| `report_bb50.py` | `bb50/` |
| `report_deploy.py`, `plot_tsne_deploy.py` | `deploy/` |
| `report_v2fix.py` | `v2fix/` |
| `report_dino_fix.py` | `dino_fix/` |
| `report_top3.py` | `top3/` |
| `report_outer5.py` | `outer5/` |
| `present_final.py` | figuras finales → `docs/figures_final/` |
| `report_shared_signal.py` | `outer5/baselines/`: por qué sin SSL y el supervisado siguen a los SSL → `docs/informe_senal_compartida.pdf` |

## `6_deploy/`: el modelo desplegable

| Script | Qué hace |
|---|---|
| `build_deploy_model.py` | Selecciona, ajusta y valida el modelo AD-vs-HC (→ `deploy/ad_mri_v2/`). |
| `predict_ad.py` | P(AD) para MRIs nuevas con un modelo de `deploy/`. |

## `legacy/`: campañas anteriores a `train.py`

Archivadas sin borrar nada; ver [`legacy/README.md`](legacy/README.md).

## `verify_local.sh`

Lint, tests, descubrimiento del dataset y una lectura del LMDB (no entrena nada).
