# 🚀 Inicio Rápido - Dataset OASIS3

## ¿Por dónde empiezo?

### Opción 1: Menú Interactivo (Recomendado)
El script más fácil de usar con interfaz de menú:

```bash
cd tools/dataset_oasis3
python scripts/quick_start.py
```

Esto te mostrará un menú interactivo donde puedes:
- Visualizar volúmenes MRI
- Explorar el dataset
- Crear galerías
- Abrir el Jupyter notebook

### Opción 2: Usar el Jupyter Notebook
Abre la guía interactiva completa:

```bash
cd tools/dataset_oasis3
jupyter notebook notebooks/Explora_Datos_OASIS3.ipynb
```

### Opción 3: Usar Scripts Directamente

#### 📊 Visualizar un volumen en 3 planos
```bash
python scripts/visualize_mri_volume.py \
    --image_path "$OASIS3_IMAGE_ROOT/sub-OAS30001ses-d0129_T1w.nii.gz" \
    --title "Mi Primera Imagen" \
    --save output.png
```

#### 🎯 Visor Interactivo con Sliders
```bash
python scripts/interactive_viewer.py \
    --image_path "$OASIS3_IMAGE_ROOT/sub-OAS30001ses-d0129_T1w.nii.gz"
```

#### 📈 Analizar el Dataset
```bash
python scripts/explore_dataset.py \
    --csv_path ../../data/dementia/mni-dataset.csv \
    --save_plot stats.png \
    --save_stats stats.json
```

#### 🎨 Crear Galería de Imágenes
```bash
python scripts/create_mri_gallery.py \
    --image_path "$OASIS3_IMAGE_ROOT/sub-OAS30001ses-d0129_T1w.nii.gz" \
    --output gallery.png \
    --style advanced \
    --title "Mi Galería"
```

## 📦 Instalación de Dependencias

Si aún no has instalado las dependencias:

```bash
python -m pip install -e "../..[toolkit]"
```

O instálalas manualmente:
```bash
pip install nibabel numpy matplotlib lmdb jupyter
```

## 📁 Estructura del Proyecto

```
Dataset-OASIS3/
├── scripts/
│   ├── quick_start.py               ← 🌟 Comienza aquí
│   ├── visualize_mri_volume.py      ← 3 planos ortogonales
│   ├── interactive_viewer.py        ← Visor con sliders
│   ├── explore_dataset.py           ← Estadísticas
│   └── create_mri_gallery.py        ← Galerías visuales
├── notebooks/
│   └── Explora_Datos_OASIS3.ipynb   ← Tutorial completo
├── data_samples/                    ← (Para guardar muestras)
├── README.md                        ← Documentación completa
├── QUICK_START.md                   ← Este archivo
└── requirements.txt                 ← Dependencias
```

## 💡 Ejemplos Rápidos

### Ejemplo 1: Ver una imagen
```bash
# Buscar un archivo
find "$OASIS3_IMAGE_ROOT" -name 'sub-OAS3*_T1w.nii.gz' -print -quit

# Visualizar (reemplaza el path)
python scripts/visualize_mri_volume.py \
    --image_path "$OASIS3_IMAGE_ROOT/sub-OAS30001ses-d0129_T1w.nii.gz"
```

### Ejemplo 2: Explorar dataset
```bash
python scripts/explore_dataset.py \
    --csv_path ../../data/dementia/mni-dataset.csv
```

### Ejemplo 3: Crear múltiples galerías
```bash
for file in "$OASIS3_IMAGE_ROOT"/sub-OAS3*_T1w.nii.gz; do
    echo "Procesando: $file"
    python scripts/create_mri_gallery.py \
        --image_path "$file" \
        --output "data_samples/$(basename ${file%.nii.gz})_gallery.png"
done
```

## 🎓 Conceptos Básicos

### ¿Qué es OASIS3?
Dataset público de imágenes MRI del cerebro con información clínica.
- ~981 sujetos
- ~1,226 sesiones
- Incluye sujetos sanos y con Alzheimer

### ¿Qué son los planos ortogonales?
- **Axial (Z):** Vista superior (de arriba hacia abajo)
- **Sagital (X):** Vista lateral (de lado)
- **Coronal (Y):** Vista frontal

### ¿Qué es normalización?
Convertir los valores de intensidad de [rango variable] a [0, 1].
Se usa para que las imágenes sean comparables entre pacientes.

## ⚡ Troubleshooting

### Error: "ModuleNotFoundError: No module named 'nibabel'"
```bash
pip install nibabel numpy matplotlib lmdb
```

### Error: "File not found"
Verifica que el archivo existe:
```bash
ls -la "$OASIS3_IMAGE_ROOT"
```

### La visualización es lenta
Intenta usar LMDB en lugar de NIfTI (ver notebook).

## 📚 Recursos

| Recurso | Ubicación |
|---------|-----------|
| **Tutorial Completo** | `notebooks/Explora_Datos_OASIS3.ipynb` |
| **Documentación** | `README.md` |
| **Scripts** | `scripts/*.py` |
| **Datasets** | `../../data/dementia/` o variables `OASIS3_*` |

## 🎯 Próximos Pasos

1. ✅ **Ejecuta** `python scripts/quick_start.py`
2. 📊 **Explora** el dataset con el analizador
3. 🎨 **Crea** galerías de imágenes
4. 📔 **Estudia** el notebook para aprender más
5. 🧠 **Investiga** patrones en los datos

---

¿Listo para comenzar? 🚀

```bash
python scripts/quick_start.py
```
