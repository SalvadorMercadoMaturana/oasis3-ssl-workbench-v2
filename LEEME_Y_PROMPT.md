# Entrega de código del repositorio T5

Este ZIP contiene **el código original del repositorio T5**, con las implementaciones de VICReg, BYOL y DINOv2 compacto, sus configuraciones Hydra y pruebas unitarias de los métodos. No contiene la adaptación previa de `t4-el7006`, datos, checkpoints ni logs.

## Prompt listo para el Codex del servidor SSH

> Tengo un repositorio remoto con datos OASIS3. Adjunto un ZIP con el código fuente del proyecto T5 donde ya funcionan VICReg, BYOL y una variante compacta de DINOv2. Lee primero `MAPA_DE_PORTABILIDAD.md`, inspecciona el repositorio remoto, sus instrucciones `AGENTS.md` y el formato real de los datos. Adapta **estas implementaciones T5** al proyecto remoto siguiendo sus convenciones, sin sustituir ni modificar los datos existentes.
>
> Conserva la composición modular de backbone, proyector, pérdida y LightningModule. Reemplaza el DataModule de BraTS por uno que lea los datos OASIS3 disponibles; por defecto usa solo OASIS3, extrae cortes 2D de un canal de los volúmenes si son 3D, y ofrece tres cortes vecinos como opción. Separa train, val y test por sujeto para evitar fuga entre sesiones. Ajusta `in_channels`, augmentaciones y configuraciones Hydra al nuevo dataset.
>
> El probe de T5 usa presencia de tumor en máscaras BraTS: no lo reutilices sin adaptación. Implementa una evaluación AD frente a HC a nivel de sujeto con encoder congelado y un backbone aleatorio como control. Verifica lectura de datos, formas de vistas, splits sin sujetos compartidos, un paso de train y val por método y recarga de checkpoints. Después ejecuta baselines completos con todos los datos admitidos y suficientes épocas, si hay recursos, y documenta métricas, comandos y limitaciones. Distingue la versión compacta incluida de DINOv2 de una reproducción completa del método.

Para entregar este ZIP al agente remoto, basta con transferirlo, extraerlo dentro de su área de trabajo y darle el prompt anterior. Las rutas relativas de los archivos reflejan el repo T5 y se conservan dentro del ZIP.
