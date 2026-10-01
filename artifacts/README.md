# Checkpoints y artefactos de modelo

Los modelos, logs y bases Optuna históricos se conservan localmente en `logs/`, `campaign_*/` y
`tuning_*/` dentro del workbench. Están excluidos de Git por tamaño y trazabilidad; sólo los
informes y figuras aprobados aparecen como candidatos versionables.

Para evaluar un checkpoint local, pasa su directorio de experimento de forma explícita:

```bash
python scripts/evaluate_frozen_probe.py logs/<experimento>
```

Una publicación reproducible debería distribuir pesos aprobados fuera del repositorio (por ejemplo,
un release o almacén de artefactos), acompañados de checksum SHA-256, configuración y licencia. No
se deben publicar checkpoints ni derivados de datos sin revisar previamente sus permisos.
