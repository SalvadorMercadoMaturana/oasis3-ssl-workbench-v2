# Mapa del código: dónde quedó cada cosa (reorganización del 2026-11-16)

La reorganización movió **sólo código y documentación**; los datos y las carpetas de
resultados (`campaign_*`, `deploy/`, `runs/`, `logs/`, CSV/JSON de la raíz) siguen donde
estaban. Ningún resultado cambió: antes de mover nada se guardaron salidas de referencia
con el código original (entrenamientos completos y mínimos, 2D y 3D, el baseline
supervisado y los 44 comandos de reportes) y se compararon con las del código nuevo.

## La estructura nueva

```text
oasis3-ssl-workbench/
├── train.py                 entrenar + evaluar un modelo SSL (el punto de entrada de las campañas)
├── src/t5_ssl/              el paquete (pip install -e .)
│   ├── recipes.py           método/variante/vistas/backbone/ROI → overrides de Hydra (sin torch)
│   ├── train.py             el pipeline de entrenamiento, en 9 pasos con nombre
│   ├── data/                LMDB/NIfTI 2D, cubos ROI 3D (roi3d.py), augmentations, GPU augment
│   ├── models/              backbones 2D/3D, cabezas, un LightningModule por método
│   ├── losses/              VICReg, BYOL, DINOv2, MAE
│   ├── callbacks/           salud de la representación (2D) y monitor de colapso (3D)
│   ├── evaluation/          NUEVO: embeddings, probes v1 y v2, métricas, identidad, QC de entrada
│   └── utils/               logging, runtime rápido (bf16, compile...), métricas de representación
├── configs/                 configuración Hydra (ver configs/README.md)
├── scripts/                 un subdirectorio por etapa (ver scripts/README.md)
│   ├── 1_data/  2_splits/  3_train/  4_evaluate/  5_reports/  6_deploy/
│   └── legacy/              campañas anteriores a train.py, archivadas (ver su README.md)
├── cloud/                   todo lo que corre en el pod de GPU (RunPod)
├── tests/                   pytest
├── tools/dataset_oasis3/    visualizadores y notebook del dataset (sin cambios)
└── docs/                    PIPELINE.md, CAMPANAS.md, este mapa, informes LaTeX/PDF, historia/
```

## Scripts: ruta vieja → ruta nueva

Todos estaban en `scripts/`. Se ejecutan igual que antes, desde la raíz del repo, con la
ruta nueva (`python scripts/5_reports/report_outer5.py ...`); cada script sigue
encontrando la raíz del proyecto por sí mismo.

| Antes (`scripts/...`) | Ahora | Etapa |
|---|---|---|
| `analyze_byol_fix.py` | `scripts/legacy/analyze_byol_fix.py` | diagnóstico BYOL 2026-10-01 |
| `analyze_byol_sensitivity.py` | `scripts/legacy/analyze_byol_sensitivity.py` | diagnóstico BYOL 2026-10-01 |
| `analyze_cv_vs_test.py` | `scripts/4_evaluate/analyze_cv_vs_test.py` | evaluación |
| `analyze_latent_split361.py` | `scripts/5_reports/analyze_latent_split361.py` | reportes |
| `analyze_ssl3d_encoders.py` | `scripts/legacy/analyze_ssl3d_encoders.py` | diagnóstico BYOL 2026-10-01 |
| `audit_oasis3_labels.py` | `scripts/1_data/audit_oasis3_labels.py` | datos |
| `bench_augment_cost.py` | `scripts/3_train/bench_augment_cost.py` | entrenamiento |
| `build_deploy_model.py` | `scripts/6_deploy/build_deploy_model.py` | despliegue |
| `build_report_metodologia.py` | `scripts/5_reports/build_report_metodologia.py` | reportes |
| `build_report_split361.py` | `scripts/5_reports/build_report_split361.py` | reportes |
| `build_report_ssl3d_v2.py` | `scripts/legacy/build_report_ssl3d_v2.py` | campaña 3D v2 |
| `build_roi_caches.py` | `scripts/1_data/build_roi_caches.py` | datos |
| `calibrate_spike_elastic.py` | `scripts/legacy/calibrate_spike_elastic.py` | calibración 3D |
| `cherry_pick_checkpoint.py` | `scripts/legacy/cherry_pick_checkpoint.py` | 2D |
| `compare_probe_runs.py` | `scripts/legacy/compare_probe_runs.py` | 2D |
| `compare_runs.py` | `scripts/4_evaluate/compare_runs.py` | evaluación |
| `evaluate_frozen_probe.py` | `scripts/legacy/evaluate_frozen_probe.py` | probe 2D (sus funciones compartidas → `t5_ssl.evaluation`) |
| `extend_heldout_test.py` | `scripts/legacy/extend_heldout_test.py` | piloto 2D |
| `extend_roi_caches.py` | `scripts/1_data/extend_roi_caches.py` | datos |
| `finalize_*.py` (8) | `scripts/legacy/finalize_*.py` | campañas 2D |
| `jobs_outer5.py` | `scripts/3_train/jobs_outer5.py` | entrenamiento |
| `make_split_361_63_84.py`, `make_split_deploy.py`, `make_split_imbalanced.py`, `make_split_outer.py` | `scripts/2_splits/` | splits |
| `measure_view_agreement.py` | `scripts/legacy/measure_view_agreement.py` | 2.5D |
| `plot_architectures_split361.py`, `plot_byol_diagnosis.py`, `plot_byol_gn.py`, `plot_byol_pair.py`, `plot_champions.py`, `plot_latent_split361.py`, `plot_split_balance.py`, `plot_ssl3d_runpod.py`, `plot_tsne_deploy.py` | `scripts/5_reports/` | reportes |
| `plot_experiments.py`, `plot_mri_views.py`, `plot_probe_label_loss.py` | `scripts/legacy/` | 2D |
| `predict_ad.py` | `scripts/6_deploy/predict_ad.py` | despliegue |
| `present_final.py` | `scripts/5_reports/present_final.py` | reportes |
| `probe_protocol.py` | **`src/t5_ssl/evaluation/protocol_v2.py`** (entró al paquete) | evaluación |
| `reeval_deploy.py` | `scripts/4_evaluate/reeval_deploy.py` | evaluación |
| `refit_pretext.py`, `tune_pretext.py`, `tune_ssl.py`, `report_pretext_tuning.py` | `scripts/legacy/` | tuning 2D |
| `render_md_tables.py` | `scripts/5_reports/render_md_tables.py` | reportes |
| `report_atrophy.py` | `scripts/legacy/report_atrophy.py` | **roto desde antes** (ver `scripts/legacy/README.md`) |
| `report_aug_recipes3d.py`, `report_bb50.py`, `report_byol_tuning.py`, `report_deploy.py`, `report_dino_fix.py`, `report_outer5.py`, `report_roi3d.py`, `report_split361_100.py`, `report_top3.py`, `report_v2fix.py` | `scripts/5_reports/` | reportes |
| `report_aug_screen.py`, `screen_augmentations.py`, `screen_aug_equal.py` | `scripts/legacy/` | screens de augmentations 2D |
| `run_roi3d.py`, `run_ssl3d.py`, `run_ssl3d_v2.py` | `scripts/legacy/` | primeras campañas 3D |
| `run_*.sh` (8) | `scripts/legacy/run_*.sh` | lanzadores de campañas 2D/locales |
| `select_representative_subjects.py` | `scripts/legacy/select_representative_subjects.py` | piloto 2D |
| `smoke_2p5d.py`, `smoke_oasis3.py` | `scripts/legacy/` | pruebas de humo 2D |
| `train_supervised.py` | `scripts/3_train/train_supervised.py` | entrenamiento |
| `tune_probe.py` | `scripts/4_evaluate/tune_probe.py` | evaluación |
| `verify_local.sh` | `scripts/verify_local.sh` (sin cambios) | verificación local |

## Archivos de la raíz

| Antes | Ahora |
|---|---|
| `train.py` | `train.py` (igual de uso; la evaluación salió a `t5_ssl.evaluation`) |
| `run_all.py`, `bench.py`, `preflight.py`, `setup.sh`, `bootstrap_runpod.sh` | `cloud/` (en el pod: `bash cloud/bootstrap_runpod.sh ...`, `python cloud/run_all.py ...`) |
| `Dockerfile` | igual (copia `cloud/setup.sh`, arranca `cloud/bootstrap_runpod.sh`) |
| `README.md` (contenía la historia de las campañas 2D) | `README.md` nuevo; el anterior en `docs/historia/README_campanas_2D.md` |
| `LEEME_Y_PROMPT.md`, `MAPA_DE_PORTABILIDAD.md`, `SESSION_HANDOFF.md` | `docs/historia/` |
| `docs/limpieza_2026-11-08.md`, `docs/runpod_byol_fix_2026-10-01.md` | `docs/historia/` |
| — | nuevos: `docs/PIPELINE.md`, `docs/CAMPANAS.md`, este mapa, `configs/README.md`, `scripts/README.md`, `scripts/legacy/README.md` |

Las rutas de scripts citadas en los docstrings, comentarios de configuración, informes
LaTeX (`docs/*.tex`), `artifacts/README.md` y `docs/ARCHITECTURE.md` se actualizaron a su
lugar nuevo; los números de los informes no cambian.

## Pruebas nuevas

| Archivo | Qué comprueba |
|---|---|
| `tests/evaluation/test_eval_metrics.py` | AUROC/AUPR rápidos (numpy) = sklearn, con empates; `mean3`; métricas a un umbral (v1 = v2); umbral F1; `auc_age`; rango efectivo; pseudo-tests reproducibles. |
| `tests/evaluation/test_input_qc.py` | Estadístico de intensidad (canal 0, cubo central) y rango aceptado (sólo sesiones OASIS3, con margen). |
| `tests/data/test_crop_cubes.py` | `crop_cubes`: orden de canales, espejo del cubo derecho, error si el cubo sale del volumen. |
| `tests/test_scripts_layout.py` | Carpetas de `scripts/`, ningún import entre etapas, nombres únicos, cada script listado en su README. |

## Funciones que estaban repetidas o escondidas en scripts de campaña

La evaluación que usa `train.py` vivía repartida entre scripts de campañas viejas
(`train.py` importaba `run_ssl3d_v2`, que importaba `screen_aug_equal`, `run_roi3d` y
`evaluate_frozen_probe`). Ahora está en el paquete, copiada sin cambios de lógica; los
scripts viejos importan desde ahí (no hay dos copias de nada).

| Función | Antes | Ahora |
|---|---|---|
| `embed` (train+val por sujeto) | `screen_aug_equal.py` | `t5_ssl.evaluation.embeddings.train_val_embeddings` |
| `split_embeddings` | `run_roi3d.py` | `t5_ssl.evaluation.embeddings.split_embeddings` |
| `external_embeddings` | `run_ssl3d_v2.py` | `t5_ssl.evaluation.embeddings.external_embeddings` |
| `concat_splits` | `evaluate_frozen_probe.py` | `t5_ssl.evaluation.embeddings.concat_splits` |
| `cv_probe` (CV 5×5) | `screen_aug_equal.py` | `t5_ssl.evaluation.probe_v1.cv_probe` |
| `evaluate` (probe train → val → test) | `evaluate_frozen_probe.py` | `t5_ssl.evaluation.probe_v1.evaluate` |
| `score` (cv + test + rango efectivo) | `run_ssl3d_v2.py` | `t5_ssl.evaluation.probe_v1.score` |
| todo el protocolo v2 | `probe_protocol.py` | `t5_ssl.evaluation.protocol_v2` (mismo contenido) |
| `pseudo_tests` | `analyze_cv_vs_test.py` | `t5_ssl.evaluation.protocol_v2.pseudo_tests` |
| `scores`, `mean3`, `fast_mean3` | `report_aug_screen.py` | `t5_ssl.evaluation.metrics` |
| `fast_auc`, `fast_ap` | `report_bb50.py` | `t5_ssl.evaluation.metrics` |
| `threshold_metrics`, `held_out_metrics` | `evaluate_frozen_probe.py` | `t5_ssl.evaluation.metrics` |
| `effective_rank` (valores singulares) | `run_ssl3d_v2.py` | `t5_ssl.evaluation.metrics.effective_rank` |
| `embed`, `views`, `scores`, `linear_cka`, `effective_rank` (varianza) | `analyze_ssl3d_encoders.py` | `t5_ssl.evaluation.identity` |
| `identity_scores` | `train.py` | `t5_ssl.evaluation.identity.identity_scores` |
| `evaluate` (todos los encoders de una corrida) | `train.py` | `t5_ssl.evaluation.run_scoring.evaluate_encoders` |
| `protocol_v2` | `train.py` | `t5_ssl.evaluation.run_scoring.score_protocol_v2` |
| `spatial_encoder`, `mean3_cv`, `mean3_test` | `train.py` | `t5_ssl.evaluation.run_scoring` |
| `epoch_metrics` | `run_ssl3d_v2.py` | `t5_ssl.evaluation.curves.epoch_metrics` |
| `QC_MARGIN`, `qc_intensity` | `predict_ad.py` | `t5_ssl.evaluation.input_qc` |
| `qc_range`, `cache_intensity` | `build_deploy_model.py` | `t5_ssl.evaluation.input_qc` |
| `crop` (cortar los cubos de un MRI) | `extend_roi_caches.py` | `t5_ssl.data.roi3d.crop_cubes` |

Ojo con los dos `effective_rank`: el de `metrics` (el que reportan los probes, sobre los
valores singulares) y el de `identity` (sobre la varianza, el de los análisis de colapso)
dan números distintos; ambos se conservaron tal cual porque cada uno ya está en reportes.

## `src/t5_ssl/train.py` por dentro

La función `train()` tenía ~360 líneas seguidas. Ahora cada etapa es una función con
nombre, llamada en el mismo orden (así los números aleatorios se consumen igual y una
semilla reproduce la misma corrida que antes):

| Paso | Función |
|---|---|
| 1 | `make_experiment_dir` |
| 2 | `build_transforms` |
| 3 | `build_datamodule` |
| 4 | `build_backbone` |
| 5 | `build_ssl_module` (usa `build_projector`, `build_loss`) |
| 6 | `build_callbacks` |
| 7 | `dm.setup` + `write_split_records` |
| 8 | `build_trainer` |
| 9 | `trainer.fit` + `last.ckpt` |
