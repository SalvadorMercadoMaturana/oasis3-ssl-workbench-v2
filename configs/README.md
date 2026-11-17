# Configuración Hydra

`train.yaml` es la configuración raíz; cada subcarpeta es un *grupo* de Hydra del que se
elige un archivo (`data=oasis3_roi3d`, `model=byol_resnet18_3d`, ...). Nadie escribe a mano
las combinaciones de las campañas 3D: `t5_ssl/recipes.py` las arma como overrides y el
`train.py` de la raíz las compone. Para ver la configuración final de una corrida sin
entrenar: `python train.py --model byol --dry-run`.

## Lo que usa el pipeline actual (3D, `train.py`)

| Grupo | Archivo | Para qué |
|---|---|---|
| `data` | `oasis3_roi3d.yaml` | Cubos ROI del cache `data/roi3d/hippo_s<roi_size>.npy` + un manifiesto de split. `recipes.roi_overrides` cambia tamaño, cache y canales. |
| `model` | `vicreg_convnext_3d.yaml`, `byol_resnet18_3d.yaml`, `dinov2_vit3d.yaml`, `pretext_convnext_3d.yaml` | Encoder 3D por defecto de cada método + cabezas. `--backbone` lo reemplaza (`recipes.BACKBONES`). |
| `loss` | `vicreg.yaml`, `byol.yaml`, `dinov2.yaml`, `pretext.yaml` | Hiperparámetros de cada objetivo. |
| `augmentation` | `roi3d_bias.yaml`, `roi3d_dinov2_bias.yaml` (+ `roi3d*.yaml`) | Vistas de las campañas; `--aug-recipe` las reemplaza campo por campo (`recipes.AUG_RECIPES`). |
| `optimizer` | `adamw.yaml`, `adamw_dinov2.yaml` | AdamW + warmup lineal + coseno; DINOv2 con weight decay creciente. |
| `trainer` | `default.yaml` (+ `full50`, `full100`, `full20`, `gpu_screen`) | Épocas, precisión, checkpoints, early stopping, health checks. |
| `perf` | `rtx4090.yaml` (`+perf=rtx4090`), `none.yaml` | Runtime rápido de GPU: bf16, TF32, AdamW fused, channels_last, compile opcional. |

## Lo que usaba la ruta 2D (`python -m t5_ssl.train`, campañas de septiembre)

`data/oasis3.yaml` (NIfTI), `oasis3_lmdb.yaml`, `oasis3_full50*.yaml`,
`oasis3_representative270.yaml`, `oasis3_2p5d*.yaml`; `model/convnext*.yaml`,
`byol_convnext*.yaml`, `byol_resnet18.yaml`, `dinov2_vit.yaml`, `vit_2p5d.yaml`;
`loss/mae*.yaml`; `augmentation/ssl_multiview*.yaml`, `dinov2_multicrop*.yaml`,
`mri_*.yaml`, `hospital_base.yaml`. Siguen funcionando (los prueba `tests/`), pero ninguna
campaña actual los usa.
