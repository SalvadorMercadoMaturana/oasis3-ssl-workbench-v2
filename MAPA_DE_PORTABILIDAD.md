# Mapa para portar los modelos T5 a otro repositorio

## Qué código corre cada método

`src/t5_ssl/train.py` es el punto de entrada Hydra. Lee `configs/train.yaml` y los grupos bajo `configs/`, construye vistas, DataModule, backbone, proyector, pérdida, LightningModule y Trainer. `src/t5_ssl/cli.py` ofrece el comando `t5-ssl train`, pero `python -m t5_ssl.train` también sirve desde la raíz con una instalación editable (`python -m pip install -e .`). El `pyproject.toml` incluido declara las dependencias. El entrenamiento original necesita BraTS-PEDs por Hugging Face o un subconjunto NPZ local; esos datos **no** están en el ZIP.

| Método | Backbone | Pérdida | Módulo de entrenamiento | Configuración principal |
|---|---|---|---|---|
| VICReg | `models/backbones/convnext.py` | `losses/vicreg.py` | `models/ssl_module.py` | `model=convnext loss=vicreg augmentation=ssl_multiview` |
| BYOL | `models/backbones/convnext.py` | `losses/byol.py` | `models/byol_module.py` | `model=byol_convnext loss=byol augmentation=ssl_multiview` |
| DINOv2 compacto | `models/backbones/vit.py` | `losses/dinov2.py` | `models/dinov2_module.py` | `model=dinov2_vit loss=dinov2 augmentation=dinov2_multicrop` |

`models/heads/projector.py` se comparte. `data/augmentations.py` produce dos vistas globales para VICReg y BYOL; para DINOv2 agrega cultivos locales. Cada vista es un `FloatTensor` `[B,C,H,W]`; la clave del batch es `image` y su valor es una tupla/lista de vistas. `C` debe coincidir con `model.backbone.in_channels`. El DataModule remoto debe preservar la misma interfaz y producir `train_dataloader` y `val_dataloader`.

El DINOv2 incluido es una variante **compacta** con distilación de CLS, predicción de parches enmascarados y KoLeo; no reproduce todos los componentes ni la escala del método completo. BYOL y DINOv2 mantienen profesores por EMA sin gradiente. VICReg utiliza dos vistas en un encoder compartido.

## Partes dependientes de BraTS que hay que reemplazar

1. `data/brats_dataset.py` lee cortes BraTS de cuatro modalidades (`t1c`, `t1n`, `t2f`, `t2w`) y deriva `label` de `seg` (presencia de tumor). Su normalización supone imágenes tipo 8/16 bits; **no la apliques ciegamente a un volumen MRI firmado**.
2. `data/brats_datamodule.py` carga `chehablab/BraTS_PED_2024` o un NPZ y divide por ID de paciente. Reemplaza carga y división por los datos reales del servidor; agrupa todas las sesiones de un sujeto en el mismo split. Para OASIS3, no mezcles ADNI3 por defecto.
3. En `train.py`, reemplaza la creación de `BraTSDataModule` y los parámetros `modalities`/`include_seg`. Ajusta `configs/data/brats2d.yaml` o crea un grupo nuevo, y cambia `in_channels` del modelo a uno para un corte T1, o a tres para cortes vecinos. El tamaño de imagen global/local debe ser compatible con el ViT y su `patch_size`.
4. `callbacks/linear_probe_callback.py` y `scripts/evaluate_frozen_probe.py` son probes **de tumor BraTS**. Usa `probe.enabled=false` hasta sustituirlos. Para AD frente a HC, extrae embeddings sin descongelar el encoder, agrega por sujeto, ajusta solo con train, escoge hiperparámetros en val y evalúa test una vez. Incluye un backbone aleatorio del mismo tipo como control.

## Verificaciones mínimas en el destino

- Inspeccionar los archivos de datos y una imagen real, y documentar forma, dtype, rango y correspondencia CSV/imagen. No asumir que rutas Windows o nombres con `(1)` serán iguales en Linux.
- Comprobar que ningún sujeto ni sesión aparece en más de un split. SSL debe entrenar solo con train, sin usar `label` en la pérdida.
- Comprobar la forma y rango de cada vista: dos globales para VICReg/BYOL; dos globales y cultivos locales para DINOv2.
- Ejecutar un paso de optimización y validación por método, verificar gradientes ausentes en ramas profesor, guardar y recargar un checkpoint. Los tests incluidos cubren parte de este contrato sobre tensores pequeños.
- Solo después lanzar experimentos completos. Un paso de prueba no constituye un baseline fiable. Guardar configuración, semilla, split, métricas y checkpoints por método.

El ZIP contiene todos los `.py` de `src/t5_ssl`, los 13 YAML de `configs`, el `pyproject.toml`, el script de probe BraTS como referencia y tests de VICReg/BYOL/DINOv2. No incluye datasets ni resultados de entrenamiento.
