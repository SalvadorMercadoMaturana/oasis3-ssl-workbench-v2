# Datos locales

Este directorio contiene el checkout local en `data/dementia` y su contrato de rutas. Los archivos
OASIS3 viven físicamente dentro del workbench, pero permanecen excluidos del historial de Git.

La configuración predeterminada espera esta disposición local:

```text
oasis3-ssl-workbench/
└── data/
    ├── README.md
    └── dementia/
        ├── mni-dataset.csv
        ├── mni-dataset.lmdb/
        │   ├── data.mdb
        │   └── lock.mdb
        ├── scripts/
        └── v0/
```

Para otra ubicación, exporta `OASIS3_DATA_ROOT`, `OASIS3_CSV`, `OASIS3_LMDB` y/o
`OASIS3_IMAGE_ROOT`. El cargador LMDB abre la base en modo de solo lectura.

Antes de publicar o compartir datos, revisa las condiciones de uso y redistribución de OASIS3.
