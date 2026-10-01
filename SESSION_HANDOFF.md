# Handoff — comparación VICReg / BYOL / DINOv2 sobre OASIS3 (2026-09-20)

Este archivo resume el estado de la sesión de trabajo del 2026-09-20 para poder
retomarla desde una nueva sesión de Claude Code abierta directamente en
`oasis3-ssl-workbench` (la sesión original se abrió en `t5_ssl_oasis3_port` y
apuntó a esta carpeta con rutas absolutas). Pégale este archivo a Claude al
empezar la nueva sesión para que recupere el contexto completo.

## Objetivo de la sesión

Diagnosticar si el entrenamiento SSL (VICReg / BYOL / DINOv2) mejora la
representación respecto a un encoder sin entrenar, para clasificación AD/HC en
OASIS3. Medidas anti-colapso "fundamentales" ya estaban implementadas
previamente (warmup+cosine LR, health-check cada 5 épocas, checkpoint
periódico, no seleccionar "best" solo por SSL loss, teacher-temp warmup y
grad-clip para DINOv2).

## Cambios de código hechos en esta sesión

- `src/t5_ssl/models/heads/projector.py`: `MLPProjector` acepta ahora
  `dropout: float = 0.0` (Dropout después de cada activación en capas
  ocultas). Default 0.0 → no afecta BYOL ni DINOv2.
- `src/t5_ssl/train.py`: pasa `dropout=proj_cfg.get("dropout", 0.0")` al
  proyector principal.
- `configs/model/convnext.yaml` (config de modelo de VICReg): agregado
  `projector.dropout: 0.1` — **ver sección de resultados para el porqué de 0.1
  y no 0.3**.
- Antes de esta sesión (contexto previo, ya en el repo): BYOL usa ResNet-18
  ImageNet-pretrained (`configs/model/byol_resnet18.yaml`), DINOv2 usa ViT-S
  canónico patch_size=60 con teacher_temp bajo (`configs/model/dinov2_vit.yaml`,
  `configs/loss/dinov2.yaml`) — ambos ajustados para replicar el informe
  EL7006 (Cossio & Carvajal).

## Corridas y resultados (30 épocas, split fijo: 615 train / 131 val / 134
test sujetos, dataset completo de 981 sujetos, manifest
`campaign_full50/train_sessions_eval_primary.csv`)

Comando base usado (variar `model.backbone.pretrained` y
`configs/model/convnext.yaml:projector.dropout` para las variantes de VICReg):

```
./.venv/Scripts/python.exe -m t5_ssl.train \
  data=oasis3_full50 trainer=full50 \
  trainer.max_epochs=30 trainer.accelerator=cpu \
  trainer.enable_progress_bar=false trainer.early_stopping_patience=null \
  data.train_cache_capacity=80 +data.eval_cache_capacity=16 \
  model.backbone.pretrained=true   # (solo para VICReg pretrained)
```

⚠️ Con 3 corridas concurrentes en esta máquina (7.7GB RAM), usar
`train_cache_capacity=80` es obligatorio — con el default (1300) crashea por
OOM. Ver `Get-CimInstance Win32_OperatingSystem` antes/durante para vigilar
memoria libre.

| Modelo | Config | test AUC (SSL) | test AUC (control script) | Nota |
|---|---|---|---|---|
| VICReg nativo | ConvNeXt-tiny, sin pretrain, sin dropout | 0.756 | 0.748 (random real) | `logs/vicreg_s25-v25-c1_convnext_tiny_2026-09-20_14-54-14` |
| VICReg + ImageNet pretrained, sin dropout | dropout=0.0 | (no evaluado con probe — se detectó overfitting antes de correr probe) | — | `logs/vicreg_s25-v25-c1_convnext_tiny_2026-09-20_17-15-32` — val_loss subía con gap acelerando (época 20: gap=8.69) |
| VICReg + ImageNet pretrained + dropout=0.3 | dropout=0.3 | **abandonado** | — | `logs/vicreg_s25-v25-c1_convnext_tiny_2026-09-20_18-06-57` — dropout=0.3 causó colapso progresivo (cos_sim 0.88→0.97, std cayendo) en vez de regularizar. Detenido en época 10. |
| **VICReg + ImageNet pretrained + dropout=0.1 (final)** | dropout=0.1 | **0.829** | 0.811* | `logs/vicreg_s25-v25-c1_convnext_tiny_2026-09-20_18-30-29` — gap train/val contenido (no repite el salto de la corrida sin dropout), salud de embeddings sana |
| BYOL (ResNet-18 pretrained) | — | **0.822** | 0.744 (random real) | `logs/byol_ema0.996_resnet18_2026-09-20_14-54-14` — mejora más clara y consistente (bootstrap P=0.991) |
| DINOv2 (ViT-S patch60) | — | 0.711 | 0.698 (random real) | `logs/dinov2_mask0.3_vit_small_2026-09-20_14-54-14` — apenas sobre azar (P=0.659) |

\* **Caveat importante sin resolver**: `scripts/evaluate_frozen_probe.py` línea
~305 construye el backbone "random" de control con `build_backbone(cfg)`
usando el `cfg` del checkpoint — es decir, si `model.backbone.pretrained=true`
en el config guardado, el control "random" también carga pesos de ImageNet en
vez de ser un encoder verdaderamente sin entrenar. Por eso el 0.811 de VICReg
pretrained no es comparable al 0.748/0.744/0.698 de las otras filas (esos sí
son random-init real). El usuario preguntó si quiero corregir esto para que el
control sea siempre random-init real — **pendiente, no implementado todavía**.

Comparando contra el baseline random-init real (0.748, de la corrida VICReg
original): VICReg pretrained+dropout=0.1 mejora +0.081, muy en línea con BYOL
(+0.078). Pero no se sabe cuánto de esa ganancia viene del pretraining de
ImageNet en sí vs. del entrenamiento SSL — el propio VICReg pretrained vs. su
"random" (ImageNet sin SSL) solo mejora +0.018 con evidencia estadística débil
(bootstrap P=0.71).

## Interpretación acordada con el usuario

- BYOL: mejora clara y sólida sobre encoder aleatorio.
- VICReg (pretrained + dropout=0.1): AUC final impresionante (0.829,
  comparable a BYOL) pero con la duda abierta de cuánto es pretraining vs. SSL.
- DINOv2: mejora marginal, no significativa — necesitaría más épocas o ajustes
  para concluir algo.
- El usuario calificó BYOL y VICReg como "impresionantes" en esta sesión.

## Pendiente / próximos pasos posibles

1. **Persistir resultados en el repo** (nada de esto está commiteado
   todavía): los JSON de `evaluate_frozen_probe.py` viven en
   `C:\Users\George_Floyd\AppData\Local\Temp\probe_full30\*.log` (temp, se
   pierden). Habíamos propuesto copiarlos a algo como
   `oasis3-ssl-workbench/reports/frozen_probe/` — sin confirmar.
2. **Corregir el control "random" del script** para que siempre use pesos
   aleatorios reales independientemente de `cfg.model.backbone.pretrained`, y
   así poder comparar limpiamente los tres VICReg entre sí.
3. **t-SNE de embeddings finales**: se generó para una corrida anterior de 270
   sujetos (no esta de 981), guardado solo en el scratchpad de esa sesión, no
   en el repo. No se ha regenerado para estas corridas de 30 épocas /
   981 sujetos.
4. Nada de esto está en git — `git status` mostrará todos estos archivos de
   config (`configs/model/convnext.yaml`, etc.) y código
   (`src/t5_ssl/models/heads/projector.py`, `src/t5_ssl/train.py`) como
   modificados, y las carpetas `logs/vicreg_*_18-*`, `logs/byol_*`,
   `logs/dinov2_*` como nuevas/sin trackear (revisar `.gitignore` — `logs/`
   probablemente está ignorado a propósito por su tamaño).

## Infraestructura / lecciones de esta sesión

- El venv correcto está en `oasis3-ssl-workbench/.venv/Scripts/python.exe` (no
  el `python` del PATH ni anaconda).
- Máquina de 7.7GB RAM: 3 procesos concurrentes de entrenamiento requieren
  `data.train_cache_capacity=80 +data.eval_cache_capacity=16` para no crashear
  por OOM (LMDB cachea volúmenes 3D completos, ~4.2MB c/u).
- `torch.set_num_threads()` explícito es necesario en `accelerator=cpu` — las
  env vars `OMP_NUM_THREADS`/`MKL_NUM_THREADS` solas no bastan en este build de
  PyTorch para Windows (ya corregido en `src/t5_ssl/train.py`).
- Para timestamps confiables usar PowerShell `Get-Date`/`Get-Process
  StartTime` o `ls -ltd --time-style=full-iso` — el reloj de bash/Python
  interno se ha visto desincronizado.
