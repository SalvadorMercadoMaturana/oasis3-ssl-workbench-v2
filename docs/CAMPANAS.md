# Campañas: qué hay en cada carpeta de resultados y qué la produjo

Las carpetas de resultados **no se movieron** en la reorganización del 2026-11-16 (sólo el
código). Aquí está, en orden cronológico, qué es cada una, con qué scripts se hizo y dónde
leer su conclusión. Los scripts marcados `legacy/` están archivados en `scripts/legacy/`
(siguen corriendo, pero no son parte del pipeline actual).

## Fase 1: cortes 2D y primeros 3D, en CPU local (2026-09)

| Carpeta | Qué fue | Scripts | Conclusión en |
|---|---|---|---|
| `campaign_100ep/` (+ `representative_oasis3*.csv/json`, `expanded_test_oasis3.*` en la raíz) | Piloto: 98 sujetos, cortes axiales 2D del LMDB, 100 épocas por método. | `legacy/select_representative_subjects.py`, `legacy/extend_heldout_test.py`, `legacy/run_100_epochs.sh`, `legacy/finalize_campaign.py`, `legacy/plot_experiments.py` | `campaign_100ep/RESULTADOS.md` |
| `tuning_pretext/`, `tuning_runs/`, `tuning_smoke*/` | Búsqueda Optuna de hiperparámetros sólo con la pérdida SSL (sin probe). | `legacy/tune_pretext.py`, `legacy/refit_pretext.py`, `legacy/report_pretext_tuning.py`, `legacy/tune_ssl.py`, `legacy/finalize_pretext_studies.py`, `legacy/plot_probe_label_loss.py` | `tuning_pretext/RESULTADOS.md` |
| `campaign_full50/` | Los 981 sujetos OASIS3, 2D, 50 épocas con early stopping. | `legacy/run_full50.sh`, `legacy/finalize_full50.py` | `campaign_full50/RESULTADOS.md` |
| `campaign_mri_fullview20/` | Vistas anatómicas de campo completo (interrumpida a pedido). | `legacy/run_mri_fullview20.sh`, `legacy/finalize_mri_fullview20.py`, `legacy/plot_mri_views.py` | `campaign_mri_fullview20/RESULTADOS.md` |
| `campaign_anatomy/` | Ablación de VICReg con vistas anatómicas. | `legacy/run_vicreg_anatomy.sh` | — |
| `campaign_2p5d_screen/` | Screening 2.5D: VICReg+cola, BYOL-GroupNorm, MAE, MAE+VICReg. | `legacy/run_2p5d_screen.sh`, `legacy/finalize_pretext_2p5d_screen.py`, `legacy/smoke_2p5d.py`, `legacy/measure_view_agreement.py` | `campaign_2p5d_screen/RESULTADOS.md` |
| `campaign_full100_gpu/`, `campaign_full30_gpu/` | Backbones actualizados, hasta 100 / 30 épocas. | `legacy/run_full100_gpu.sh`, `legacy/run_full100_local_sequential.sh`, `legacy/run_full30_local_sequential.sh`, `legacy/finalize_full100_gpu.py` | `campaign_full100_gpu/RESULTADOS.md` |
| `campaign_full50_pretrained/`, `campaign_full50_pretrained_balanced/`, `campaign_full50_balanced/` | Backbones ImageNet, 50 épocas; split balanceado AD/HC. **El manifiesto balanceado sigue siendo el default de `t5_ssl.recipes.MANIFEST`.** | `legacy/finalize_full50_pretrained.py`, `legacy/finalize_full50_pretrained_balanced.py` | sus `RESULTADOS.md` |
| `campaign_aug_screen/` | Screen de augmentations tipo hospital (BYOL 2D, 30 épocas). | `legacy/screen_augmentations.py`, `legacy/report_aug_screen.py` | `RESULTADOS.md` |
| `campaign_aug_screen2/` | Screen 2: artefactos a igual intensidad (BYOL 2D, 10 épocas). | `legacy/screen_aug_equal.py` | `RESULTADOS.md` |
| `campaign_roi3d/` | ROI hipocampal 3D vs cortes 2D; calibración de artefactos 3D (`calibration_s32.json`). | `legacy/run_roi3d.py`, `legacy/calibrate_spike_elastic.py` | `RESULTADOS.md` |
| `campaign_ssl3d/` | VICReg, BYOL y DINOv2 en 3D (ROI hipocampal). | `legacy/run_ssl3d.py` | `RESULTADOS.md` |
| `campaign_ssl3d_v2/` | Segunda pasada 3D tras el diagnóstico de DINOv2 y BYOL (2026-09-30). | `legacy/run_ssl3d_v2.py`, `legacy/build_report_ssl3d_v2.py` (→ `docs/informe_ssl3d_v2.pdf`), `legacy/analyze_ssl3d_encoders.py`, `legacy/analyze_byol_sensitivity.py` | `RESULTADOS.md`, informe PDF |

`logs/` guarda los experimentos Lightning de esta fase (`python -m t5_ssl.train`).
`backup_pre_hippo_pipeline_2026-09-21/` y `backup_pre_ssl3d_fix_2026-09-30/` son copias
de configuraciones y código antes de dos cambios grandes (cada uno con su `README.md`).

## Fase 2: `train.py` en GPU (RunPod), todo dentro de `campaign_ssl3d_runpod/` (2026-10 y 2026-11)

Todas las corridas son de `train.py` (directo, con `cloud/run_all.py` o con
`cloud/queue_runner.py`) salvo los baselines supervisados (`scripts/3_train/train_supervised.py`).

| Subcarpeta | Qué fue | Scripts de análisis / reporte | Conclusión en |
|---|---|---|---|
| `screen30/` | Arreglo de BYOL (relleno de borde, cabezas anchas/GN), 30 épocas, 3 semillas. | `legacy/analyze_byol_fix.py`, `4_evaluate/compare_runs.py` | `ANALYSIS.md`, `COMPARISONS.md` |
| `final50/` | Los tres métodos a 50 épocas con el runtime rápido. | `4_evaluate/compare_runs.py`, `5_reports/plot_ssl3d_runpod.py` | `COMPARISONS.md`, `RESULTADOS.md` (raíz de la campaña) |
| `dropout20/`, `augs20/` | Dropout de BYOL; 19 recetas de vistas × 3 métodos. | `5_reports/report_aug_recipes3d.py`, `5_reports/render_md_tables.py` | `REPORT.md`, `TABLA_*.md` |
| `roi50/` | Tamaño y número de regiones (32³, 48³, 64³, multi3). | `5_reports/report_roi3d.py` | `REPORT.md` |
| `champions50/` | Mejor receta + mejor ROI por método, 50 épocas. | `5_reports/plot_champions.py` | `DELTAS.md` |
| `split_361_63_84/`, `split361_30/`, `latent_split361/` | Nuevo split 361/63/84, 30 épocas; espacio latente. | `2_splits/make_split_361_63_84.py`, `5_reports/plot_champions.py --runs split361_30`, `5_reports/analyze_latent_split361.py`, `5_reports/plot_latent_split361.py`, `5_reports/plot_architectures_split361.py`, `5_reports/plot_split_balance.py`, `5_reports/build_report_split361.py` (→ `docs/informe_split361.pdf`), `5_reports/build_report_metodologia.py` (→ `docs/informe_metodologia_split361.pdf`) | informes PDF |
| `byol_diag30/`, `byol_tune30/`, `byol_fbn30/` | Por qué sube la pérdida de validación de BYOL; tuning GroupNorm; BN congelada. | `5_reports/plot_byol_diagnosis.py`, `5_reports/report_byol_tuning.py`, `5_reports/plot_byol_gn.py`, `5_reports/plot_byol_pair.py` | `DIAGNOSTICO.md`, `REPORT.md`, `BYOL_FBN.md` |
| `split361_100/`, `latent_split361_100/` | Campeones a 100 épocas vs supervisado vs sin SSL. | `5_reports/report_split361_100.py` | `REPORT.md` |
| `bb50/`, `split_361_imb/`, `split_agematch/` | Mismo backbone para los tres métodos; test 1:6; split emparejado por edad. | `2_splits/make_split_imbalanced.py`, `5_reports/report_bb50.py` | `REPORT.md` |
| `split_deploy/`, `split_deploy_adni/`, `deploy/runs/dep50_*` | Split de despliegue (test representativo + ADNI externo) y primer modelo desplegado (`deploy/ad_mri_v1`). | `2_splits/make_split_deploy.py`, `5_reports/report_deploy.py`, `5_reports/plot_tsne_deploy.py`, `6_deploy/build_deploy_model.py` | `deploy/runs/REPORT.md`, `deploy/ad_mri_v1/MODEL_CARD.md` |
| `atrophy/` | Cinco regiones de atrofia apiladas (abortada: peor que el hipocampo). | `legacy/report_atrophy.py` (roto, ver `scripts/legacy/README.md`) | — |
| `v2fix/`, `deploy/runs/v2_*` | Protocolo de evaluación v2, validación SSL aumentada, BYOL con cabezas BN → **`deploy/ad_mri_v2`**. | `4_evaluate/reeval_deploy.py`, `5_reports/report_v2fix.py`, `6_deploy/build_deploy_model.py` | `REPORT.md`, `deploy/ad_mri_v2/MODEL_CARD.md` |
| `dino_fix/`, `deploy/runs/dino_fix*` | Por qué DINOv2 rinde menos (prototipos, lectura del probe, ViT vs CNN, entrada). | `5_reports/report_dino_fix.py` | `REPORT.md` |
| `top3/` | Los tres mejores de cada método (protocolo v2). | `5_reports/report_top3.py` | `REPORT.md` |
| `split_study/`, `split_outer5/`, `outer5/` | CV vs test; tareas pretexto (rotación/jigsaw); tuning; validación externa de 5 pliegues (2026-11-16). | `4_evaluate/analyze_cv_vs_test.py`, `2_splits/make_split_outer.py`, `3_train/jobs_outer5.py`, `4_evaluate/tune_probe.py`, `5_reports/report_outer5.py`, `5_reports/present_final.py` (→ `docs/figures_final/`) | `outer5/REPORT.md` |
| `outer5/baselines/` | Por qué sin SSL y el supervisado se mueven con los SSL: líneas base triviales (volumen hipocampal, estadísticas de intensidad, pesos aleatorios) con el protocolo v2 (2026-11-21). | `4_evaluate/trivial_baselines.py`, `5_reports/report_shared_signal.py` (→ `docs/informe_senal_compartida.pdf`) | `summary.json`, informe PDF |
| `ingest/` | Tuning de ingesta y backbone para los 4 métodos (2026-11-22/23): normalización, hemisferios por separado, remuestreo, MedicalNet, cortes 2D del hipocampo, ConvNeXt-T con pesos ImageNet-22k; pliegue 0 de `split_deploy`. | `3_train/jobs_ingest.py`, `5_reports/report_ingest.py` | `REPORT.md`, `summary.json` |
| `brain2d/` | Cerebro completo como 3 planos 2D (axial, coronal, sagital) por el centro de masa, z-score sobre el tejido; probe de varios cortes, vistas de 3 cortes, lr ×0.1 (2026-11-22). **Descartado**: el SSL queda bajo el encoder congelado y bajo el 3D. | `1_data/build_brain2d_cache.py`, `3_train/jobs_ingest.py --roi brain2d`, `4_evaluate/reeval_brain2d.py`, `5_reports/report_brain2d.py` | `REPORT.md`, `summary.json` |
| `bench/`, `bench_dinov2/` | Throughput del runtime rápido. | `cloud/bench.py` | `BENCH.md` |

## Otras carpetas de la raíz

| Carpeta | Qué contiene |
|---|---|
| `data/` | Datos privados (LMDB, CSV, caches de ROI). Ignorada por Git. |
| `deploy/` | Modelos desplegables (`ad_mri_v1`, `ad_mri_v2`): pesos, probes, umbrales, `MODEL_CARD.md`. |
| `runs/` | Corridas locales de `train.py` (default de `--out-root`). |
| `logs/` | Experimentos Lightning de la ruta 2D (`python -m t5_ssl.train`). |
| `outputs/diagnostics/` | Diagnósticos de entradas SSL de septiembre. |
| `reports/`, `figures_smoke/`, `artifacts/` | Reportes sueltos, figuras de pruebas de humo y el contrato de checkpoints externos. |
| `smoke_results.json`, `smoke_2p5d_results.json` | Salidas de `legacy/smoke_oasis3.py` y `legacy/smoke_2p5d.py`. |
| `cohort_representativeness.json`, `representative_data_check.json` | Chequeos de la cohorte del piloto de 98 sujetos. |
