# Scripts archivados (campañas anteriores a `train.py`)

Todo lo que está aquí produjo resultados que siguen en el repositorio, pero ya no es parte del
pipeline actual ([`docs/PIPELINE.md`](../../docs/PIPELINE.md)). Se archivaron el 2026-11-16 sin
borrar nada; sólo se ajustaron rutas e imports. Se ejecutan desde la raíz del repositorio
(`python scripts/legacy/<script>.py`) y sólo importan entre sí o del paquete `t5_ssl`. Qué
carpeta de resultados produjo cada uno: [`docs/CAMPANAS.md`](../../docs/CAMPANAS.md).

Varias funciones de estos scripts se movieron al paquete porque el pipeline actual las usa
(`t5_ssl.evaluation`); los scripts las vuelven a importar con el nombre antiguo, así que
`import run_ssl3d_v2 as v2; v2.score(...)` y similares siguen funcionando.

## Cortes 2D del LMDB, en CPU (septiembre de 2026)

| Script | Qué hace | Resultados |
|---|---|---|
| `select_representative_subjects.py` | Cohorte reproducible de sujetos OASIS3 (sin copiar imágenes). | `representative_oasis3.*`, `campaign_full50/all_sessions.csv` |
| `extend_heldout_test.py` | Test ampliado predeclarado sin tocar train/val del piloto. | `expanded_test_oasis3.*` |
| `smoke_oasis3.py` | Un batch real de train/val y recarga de checkpoint por método. | `smoke_results.json` |
| `run_100_epochs.sh`, `finalize_campaign.py` | Piloto de 98 sujetos, 100 épocas por método, probe AD/HC. | `campaign_100ep/` |
| `plot_experiments.py` | Curvas SSL por método y métricas del probe. | `campaign_100ep/figures/` |
| `evaluate_frozen_probe.py` | Probe AD vs HC por sujeto con el encoder congelado (+ control aleatorio). Lo usan los `finalize_*`. | `ad_hc_subject_probe.json` de cada experimento |
| `compare_probe_runs.py` | Bootstrap pareado entre dos corridas del probe. | — |
| `cherry_pick_checkpoint.py` | Elige el mejor checkpoint periódico de un experimento. | — |
| `tune_ssl.py` | Optuna reanudable, seleccionando en sujetos de validación. | `tuning_runs/`, `tuning_smoke*/` |
| `tune_pretext.py`, `refit_pretext.py`, `report_pretext_tuning.py`, `finalize_pretext_studies.py` | Tuning sólo con la pérdida SSL; reajuste a 100 épocas; reporte. | `tuning_pretext/` |
| `plot_probe_label_loss.py` | Log-loss del probe frente a baselines de imagen y prevalencia. | `tuning_pretext/` |
| `run_full50.sh`, `finalize_full50.py` | Los 981 sujetos, 50 épocas con early stopping. | `campaign_full50/` |
| `run_mri_fullview20.sh`, `finalize_mri_fullview20.py`, `plot_mri_views.py` | Vistas de campo completo (interrumpida). | `campaign_mri_fullview20/` |
| `run_vicreg_anatomy.sh` | Ablación de VICReg con vistas anatómicas. | `campaign_anatomy/` |
| `run_2p5d_screen.sh`, `finalize_pretext_2p5d_screen.py`, `smoke_2p5d.py`, `measure_view_agreement.py` | Screening 2.5D (VICReg+cola, BYOL-GN, MAE). | `campaign_2p5d_screen/`, `smoke_2p5d_results.json` |
| `run_full100_gpu.sh`, `run_full100_local_sequential.sh`, `run_full30_local_sequential.sh`, `finalize_full100_gpu.py` | Backbones actualizados, 100 / 30 épocas. | `campaign_full100_gpu/`, `campaign_full30_gpu/` |
| `finalize_full50_pretrained.py`, `finalize_full50_pretrained_balanced.py` | Backbones ImageNet; split balanceado AD/HC. | `campaign_full50_pretrained*/` |
| `screen_augmentations.py`, `report_aug_screen.py` | Screen 1 de artefactos de adquisición (BYOL 2D). `screen_augmentations` define `ROOT`, `MANIFEST` y `to_hydra`, que importan los demás scripts de esta carpeta. | `campaign_aug_screen/` |
| `screen_aug_equal.py` | Screen 2: artefactos a igual intensidad, 10 épocas. | `campaign_aug_screen2/` |

## Primeros 3D, en CPU (fines de septiembre de 2026)

| Script | Qué hace | Resultados |
|---|---|---|
| `run_roi3d.py`, `calibrate_spike_elastic.py` | ROI hipocampal 3D vs corte 2D; calibración de artefactos 3D. | `campaign_roi3d/` |
| `run_ssl3d.py` | VICReg, BYOL y DINOv2 en 3D bajo un protocolo corto. | `campaign_ssl3d/` |
| `run_ssl3d_v2.py` | Segunda pasada tras el diagnóstico del 2026-09-30. **Su protocolo de probe (v1) es el que `train.py` reproduce**, ahora desde `t5_ssl.evaluation.probe_v1`. | `campaign_ssl3d_v2/` |
| `build_report_ssl3d_v2.py` | Figuras y tablas de `docs/informe_ssl3d_v2.tex`. | `docs/` |
| `analyze_ssl3d_encoders.py`, `analyze_byol_sensitivity.py` | Qué hace el SSL a la representación; qué augmentación domina BYOL. | `campaign_ssl3d_v2/` |
| `analyze_byol_fix.py` | Identidad y sensibilidad a augmentaciones en el screen del arreglo de BYOL (ya con `train.py`). | `campaign_ssl3d_runpod/screen30/ANALYSIS.md` |

## Roto

`report_atrophy.py` (campaña `campaign_ssl3d_runpod/atrophy/`, abortada) ya no funcionaba
antes de la reorganización: llama a `build_deploy_model.ensemble_oof`, que se eliminó de ese
script cuando se rehízo el modelo desplegable. Se conserva como registro de lo que se midió.
