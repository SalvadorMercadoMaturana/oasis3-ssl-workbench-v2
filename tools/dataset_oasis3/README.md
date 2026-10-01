# Dataset-OASIS3 🧠

Toolkit completo para visualizar, explorar y analizar datos de resonancia magnética del dataset OASIS3.

En el workbench unificado, ejecuta estos comandos desde `tools/dataset_oasis3/` o usa las rutas
completas desde la raíz. El dataset privado se descubre mediante las variables `OASIS3_*` o en
`../../data/dementia`.

## 📋 Contenido

```
Dataset-OASIS3/
├── scripts/                          # Scripts Python para visualización
│   ├── visualize_mri_volume.py      # Visualizar volumen en 3 planos
│   ├── interactive_viewer.py        # Visor interactivo con sliders
│   ├── explore_dataset.py           # Estadísticas del dataset
│   └── create_mri_gallery.py        # Crear galerías de rebanadas
├── notebooks/
│   └── Explora_Datos_OASIS3.ipynb   # Guía completa interactiva
└── README.md                         # Este archivo
```

## 🚀 Inicio Rápido

### 1. Visualizar un volumen MRI individual

```bash
python scripts/visualize_mri_volume.py \
    --image_path /path/to/volume.nii.gz \
    --title "Mi Primera Imagen MRI" \
    --save output.png
```

**Salida:** Vista en 3 planos ortogonales (Axial, Sagital, Coronal)

### 2. Explorador interactivo con sliders

```bash
python scripts/interactive_viewer.py \
    --image_path /path/to/volume.nii.gz
```

**Interacción:** Usa los sliders para navegar por cada plano

### 3. Analizar estadísticas del dataset

```bash
python scripts/explore_dataset.py \
    --csv_path ../../data/dementia/mni-dataset.csv \
    --save_plot stats_plot.png \
    --save_stats stats.json
```

**Salida:** Gráficos de distribución de grupos, sesiones, etc.

### 4. Crear galería visual

```bash
python scripts/create_mri_gallery.py \
    --image_path /path/to/volume.nii.gz \
    --output gallery.png \
    --style advanced \
    --title "Mi Galería"
```

**Estilos:** `simple` (3×3), `advanced` (3×4 con descripción anatómica)

## 📔 Jupyter Notebook

Abre **`notebooks/Explora_Datos_OASIS3.ipynb`** para:

- ✅ Introducción a OASIS3 y las imágenes MRI
- ✅ Tutorial paso a paso con ejemplos
- ✅ Carga de datos desde LMDB y NIfTI
- ✅ Visualizaciones interactivas
- ✅ Análisis estadístico del dataset
- ✅ Ejercicios prácticos

```bash
jupyter notebook notebooks/Explora_Datos_OASIS3.ipynb
```

## 📊 Estructura de Datos

### CSV Clínico
```
/home/datasets/dementia/mni-dataset.csv
├── study_name: "OASIS3"
├── original_code: "sub-OAS30001ses-d0129"  # Código de sesión
├── subject: "sub-OAS30001"                 # ID del sujeto
├── group: "AD" | "HC" | "MCI" | ...        # Grupo clínico
└── idx: "00001"                            # Índice único
```

### Imágenes NIfTI
```
/mnt/home/shared/oasis3/
└── sub-OAS30001ses-d0129/
    └── sub-OAS30001ses-d0129_T1w.nii.gz  # Imagen T1 ponderada
```

### Base de Datos LMDB
```
/home/datasets/dementia/mni-dataset.lmdb
├── data.mdb              # Datos serializados (pickle)
└── lock.mdb              # Archivo de bloqueo
```

## 🔧 Requisitos

```
nibabel>=4.0.0        # Lectura de archivos NIfTI
numpy>=1.20.0         # Operaciones numéricas
matplotlib>=3.3.0     # Visualización
lmdb>=1.0.0           # Acceso a base de datos LMDB
```

Instalar:
```bash
pip install nibabel numpy matplotlib lmdb
```

### Instalación local reproducible

En WSL, desde la raíz del repositorio:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/quick_start.py --check
.venv/bin/python scripts/quick_start.py
```

Los datos copiados localmente están en `../../data/dementia`. Por ejemplo:

```bash
.venv/bin/python scripts/explore_dataset.py \
    --csv_path ../../data/dementia/mni-dataset.csv \
    --save_plot /tmp/oasis3_stats.png
```

`.gitignore` evita añadir entornos, LMDB, NIfTI y resultados generados al historial de Git.

## 💡 Conceptos Clave

### Volumen MRI
- **Formato:** 3D array de intensidades
- **Típicamente:** 256×256×80 voxels
- **Unidades:** Milímetros (escala: ~1mm por voxel)

### Planos Ortogonales
- **Axial:** Vista superior (transversal) - Z constante
- **Sagital:** Vista lateral - X constante  
- **Coronal:** Vista frontal - Y constante

### Normalización
- Se usa percentiles robustos (1%, 99%) para mapear a [0, 1]
- Esto maneja variaciones en intensidad entre máquinas y pacientes

### Rebanadas (Slices)
- Se extraen cortes 2D individuales del volumen 3D
- Típicamente 1-5 rebanadas equidistantes por volumen
- Se pueden usar canales múltiples (1 o 3 vecinos)

## 📈 Estadísticas Típicas

| Métrica | Valor |
|---------|-------|
| Total de sujetos OASIS3 | ~981 |
| Total de sesiones | ~1,226 |
| Sujetos AD (Alzheimer's) | ~350 |
| Sujetos HC (Healthy Control) | ~250 |
| Sesiones por sujeto (media) | 1.25 |
| Dimensión típica | 256×256×80 |

## 🎓 Ejercicios Prácticos

### Ejercicio 1: Identificar anomalías
1. Carga un volumen con `visualize_mri_volume.py`
2. Observa los 3 planos
3. ¿Ves algo inusual? (ventrículo dilatado, lesiones, etc.)

### Ejercicio 2: Comparar sujetos
```bash
# Carga múltiples volúmenes
for file in /mnt/home/shared/oasis3/sub-OAS3*/sub-OAS3*_T1w.nii.gz; do
    python scripts/create_mri_gallery.py --image_path $file --output ${file%.nii.gz}_gallery.png
done
```

### Ejercicio 3: Análisis estadístico
```bash
python scripts/explore_dataset.py \
    --csv_path /home/datasets/dementia/mni-dataset.csv
```

Responde:
- ¿Cuál es la proporción AD/HC?
- ¿Cuántas sesiones por sujeto en promedio?

## 🔍 Troubleshooting

### Error: "File not found"
```bash
# Verifica que el archivo existe
ls -la /path/to/file.nii.gz
```

### Error: "Cannot read NIfTI file"
```bash
# Asegúrate de que el archivo es válido
file /path/to/file.nii.gz  # Debe ser gzip
```

### Visualización lenta
```bash
# Reduce el tamaño del volumen o usa LMDB en lugar de NIfTI
# Ver oasis3_lmdb_datamodule.py para mejor rendimiento
```

## 📚 Referencias

- **OASIS3 Dataset:** https://www.oasis-brains.org/
- **NIfTI Format:** https://nifti.nimh.nih.gov/
- **Neuroimaging Basics:** https://neuroimaging.astroml.org/

## 📝 Notas

- Las imágenes están en formato MNI (sistema de coordenadas estándar)
- Los volúmenes se convierten a formato canónico RAS (Right-Anterior-Superior)
- Normalización robusta: percentiles 1% y 99% para manejar outliers

---

¡Comienza a explorar los datos con el notebook `Explora_Datos_OASIS3.ipynb`! 🚀
