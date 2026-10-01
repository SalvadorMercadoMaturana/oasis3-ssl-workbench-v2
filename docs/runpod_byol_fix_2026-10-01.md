# BYOL corregido y entrenamiento de 50 épocas en GPU (2026-10-01)

Métricas agregadas (media de AUROC, F1-macro y AUPR del probe congelado, protocolo
de `campaign_ssl3d_v2`: CV 5×5 en 357 sujetos AD/HC de train+val y probe en 56 de test).
Δ con bootstrap pareado sobre sujetos (`scripts/compare_runs.py`). Las salidas
completas quedan en `runs/pod/` (gitignorado).

## Hardware y costo

RunPod, RTX 5090 community (32 GB, 20 vCPU, 0.69 USD/h): no había RTX 4090
disponible y era la siguiente opción bajo 0.8 USD/h. PyTorch 2.8 + CUDA 12.8
(la imagen traía 2.4, sin kernels sm_120). 45 min de pod en total (instalación,
bench, grilla de 24 corridas, análisis y final de 12 trabajos): **0.49 USD**.

## Rendimiento

Bench, un trabajo, 100 pasos con batch 16 (img/s = volúmenes por segundo en el encoder online):

| Modelo | Runtime de las campañas | Rápido (bf16, TF32, workers, channels_last, AdamW fused) | + torch.compile |
|---|---:|---:|---:|
| BYOL ResNet-18 3D | 181 | 443 (2.4×) | 169 |
| VICReg ConvNeXt-T 3D | 140 | 437 (3.1×) | 156 |
| DINOv2 ViT-S 3D | 217 | 323 (1.5×) | 131 |

`torch.compile` queda opcional (`--compile on`): con cubos de 32³ y batch 16 es 2.5–2.8× más lento.

Final de 50 épocas, 4 trabajos a la vez con CUDA MPS (12 trabajos, 7.7 min de reloj):

| Modelo | img/s por trabajo | s/época | VRAM pico por proceso | min por corrida (con probes) |
|---|---:|---:|---:|---:|
| BYOL `border_wide` | 681 | 1.2 | 1.19 GB | 2.4 |
| VICReg | 444 | 1.8 | 1.07 GB | 2.4 |
| DINOv2 | 355 | 2.3 | 1.25 GB | 2.6 |

GPU completa: 7.2 de 32 GB en el pico; utilización media 41 % (el cuello es la CPU de las augmentations).

## Experimentos de BYOL (30 épocas, 3 semillas)

| Variante | CV | Test | entre/dentro (online) | sensibilidad vistas (online) |
|---|---:|---:|---:|---:|
| campaña (relleno con ceros) | 0.810 | 0.759 | 3e-5 – 0.23 | 4 – 37 000 |
| `border` | 0.813 | 0.812 | 2e-5 – 4.6 | 0.2 – 45 000 |
| `border_wide` (cabezas 4096) | **0.827** | 0.831 | 1.4 – 2.2 | 0.5 – 0.7 |
| `border_gn` (GroupNorm) | 0.801 | 0.823 | 5e-4 – 0.39 | 2.6 – 2 100 |
| `border_predlr` (LR predictor ×10) | 0.823 | 0.821 | 1.3 – 5.5 | 0.2 – 0.7 |
| `border_h3` (las tres) | 0.820 | 0.835 | 7.0 – 8.6 | 0.1 |
| VICReg (referencia) | 0.838 | 0.867 | 3.7 – 4.3 | 0.2 – 0.3 |

- El relleno de borde solo **no** basta: el encoder online colapsa sobre otro atajo en 2 de 3 semillas.
- Lo que evita el colapso son las cabezas: con proyector/predictor de 4096 o LR del predictor ×10 los sujetos quedan separados (cociente ≥ 1.3, sensibilidad < 1).
- Elegido por CV: `border_wide`. Frente a BYOL de la campaña: CV +0.018 [+0.001, +0.034], test +0.072 [−0.015, +0.155].

## Final de 50 épocas (3 semillas)

| Modelo | CV | Test | Δ vs su inicio sin SSL: CV | Δ test |
|---|---:|---:|---:|---:|
| VICReg | 0.841 | 0.887 | +0.029 [+0.009, +0.050] | +0.062 [−0.009, +0.145] |
| BYOL `border_wide` | 0.820 | 0.810 | +0.011 [−0.025, +0.039] | −0.032 [−0.101, +0.042] |
| DINOv2 | 0.787 | 0.814 | +0.038 [−0.000, +0.078] | +0.009 [−0.094, +0.105] |

- VICReg sigue siendo el mejor y es el único con aporte del SSL claro en CV.
- BYOL ya no colapsa, pero a 50 épocas no supera a su inicio ImageNet: la representación es sana pero el probe lineal no gana.
- Probe espacial (mapa previo al GAP, 2³ celdas): no sube el techo de las CNN (VICReg 0.820 / 0.851, BYOL 0.812 / 0.816); en DINOv2 los tokens de parche dan más que el CLS en CV (0.816 frente a 0.787).
