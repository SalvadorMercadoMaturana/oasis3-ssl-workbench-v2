# 📑 Índice de Archivos - Dataset-OASIS3

## 📁 Estructura Completa

```
Dataset-OASIS3/
├── scripts/                              # Scripts Python ejecutables
│   ├── quick_start.py                   # ⭐ Inicio rápido con menú interactivo
│   ├── visualize_mri_volume.py          # Visualizar volumen en 3 planos
│   ├── interactive_viewer.py            # Visor interactivo con sliders
│   ├── explore_dataset.py               # Análisis estadístico del dataset
│   └── create_mri_gallery.py            # Crear galerías de rebanadas
│
├── notebooks/                            # Jupyter notebooks
│   └── Explora_Datos_OASIS3.ipynb      # 📔 Tutorial completo interactivo
│
├── data_samples/                         # Carpeta para guardar muestras
│
├── 📄 README.md                          # Documentación completa
├── 🚀 QUICK_START.md                     # Guía de inicio rápido
├── 📑 INDEX.md                           # Este archivo
├── requirements.txt                      # Dependencias Python
└── 📋 Este directorio                    # Herramientas OASIS3

```

## 🎯 Qué Contiene Cada Archivo

### 📄 **README.md**
- Descripción completa del proyecto
- Guía de uso de cada script
- Conceptos clave explicados
- Estadísticas del dataset
- Troubleshooting

### 🚀 **QUICK_START.md**
- Inicio rápido en 3 líneas
- Ejemplos de código
- Instalación de dependencias
- Conceptos básicos

### 📑 **INDEX.md**
- Este archivo
- Descripción de todos los archivos
- Guía de navegación

### requirements.txt
- Lista de dependencias Python
- Instala con: `pip install -r requirements.txt`

---

## 🛠️ Scripts (Carpeta `scripts/`)

### ⭐ **quick_start.py**
**Uso:** `python scripts/quick_start.py`

Menú interactivo que te guía a través de todas las funcionalidades:
- Visualizar volúmenes
- Explorar dataset
- Crear galerías
- Abrir notebook

**Ideal para:** Usuarios nuevos

---

### 📊 **visualize_mri_volume.py**
**Uso:** 
```bash
python scripts/visualize_mri_volume.py \
    --image_path /path/to/volume.nii.gz \
    --title "Mi Imagen" \
    --save output.png
```

**Función:**
- Carga un archivo NIfTI
- Normaliza la imagen
- Muestra los 3 planos ortogonales:
  - Axial (Z): vista superior
  - Sagital (X): vista lateral
  - Coronal (Y): vista frontal
- Imprime estadísticas

**Parámetros:**
- `--image_path` (obligatorio): Ruta al archivo NIfTI
- `--title` (opcional): Título de la visualización
- `--save` (opcional): Guardar como PNG

**Salida:** Gráfico con 3 subplots + información

---

### 🎯 **interactive_viewer.py**
**Uso:**
```bash
python scripts/interactive_viewer.py \
    --image_path /path/to/volume.nii.gz \
    --title "Mi Visor"
```

**Función:**
- Visor interactivo con sliders
- Navega por los 3 planos simultáneamente
- Muestra información en tiempo real
- Controles deslizantes para cada eje (X, Y, Z)

**Parámetros:**
- `--image_path` (obligatorio): Ruta al archivo NIfTI
- `--title` (opcional): Título de la ventana

**Salida:** Ventana interactiva con 3 vistas + sliders

---

### 📈 **explore_dataset.py**
**Uso:**
```bash
python scripts/explore_dataset.py \
    --csv_path ../../data/dementia/mni-dataset.csv \
    --save_plot stats_plot.png \
    --save_stats stats.json
```

**Función:**
- Lee el CSV clínico
- Calcula estadísticas del dataset
- Muestra distribuciones de grupos (AD/HC)
- Crea visualizaciones de:
  - Distribución de grupos clínicos
  - Registros por sujeto
  - Sesiones por sujeto
  - Estadísticas resumidas

**Parámetros:**
- `--csv_path` (obligatorio): Ruta al CSV
- `--save_plot` (opcional): Guardar gráficos como PNG
- `--save_stats` (opcional): Guardar JSON con estadísticas

**Salida:** Gráficos + JSON con estadísticas

---

### 🎨 **create_mri_gallery.py**
**Uso:**
```bash
python scripts/create_mri_gallery.py \
    --image_path /path/to/volume.nii.gz \
    --output gallery.png \
    --style advanced \
    --title "Mi Galería"
```

**Función:**
- Extrae rebanadas equidistantes de un volumen
- Crea una composición visual (galería)
- Dos estilos disponibles:
  - `simple`: 3×3 rebanadas
  - `advanced`: 3×4 rebanadas con descripción anatómica

**Parámetros:**
- `--image_path` (obligatorio): Ruta al archivo NIfTI
- `--output` (obligatorio): Ruta de salida (PNG)
- `--style` (opcional): "simple" o "advanced"
- `--title` (opcional): Título de la galería

**Salida:** Imagen PNG con galería de rebanadas

---

## 📔 Jupyter Notebook

### **notebooks/Explora_Datos_OASIS3.ipynb**
**Uso:** `jupyter notebook notebooks/Explora_Datos_OASIS3.ipynb`

**Contenido:**
1. Instalación de dependencias
2. Carga de librerías
3. Configuración de rutas
4. Funciones auxiliares
5. Carga del CSV clínico
6. Análisis estadístico
7. Visualización de distribuciones
8. Carga de volúmenes NIfTI
9. Visualización en 3 planos
10. Extracción de rebanadas
11. Análisis de intensidades
12. Exploración de LMDB
13. Carga desde LMDB
14. Ejercicios prácticos
15. Conclusiones y próximos pasos

**Ideal para:** Aprendizaje interactivo y exploración detallada

---

## 📂 Carpeta data_samples/

Carpeta vacía para guardar tus propias muestras:
- Galerías generadas
- Imágenes exportadas
- Datos procesados

---

## 🔗 Relaciones Entre Archivos

```
QUICK_START.md (inicio)
    ↓
    ├─→ quick_start.py (menú interactivo)
    │      ↓
    │      ├─→ visualize_mri_volume.py
    │      ├─→ interactive_viewer.py
    │      ├─→ explore_dataset.py
    │      ├─→ create_mri_gallery.py
    │      └─→ Explora_Datos_OASIS3.ipynb
    │
    └─→ README.md (documentación completa)

```

---

## 📊 Datos que Utilizan

### Archivos Utilizados:
- `../../data/dementia/mni-dataset.csv` - Metadatos clínicos
- `$OASIS3_IMAGE_ROOT` - Imágenes NIfTI, cuando estén disponibles
- `../../data/dementia/mni-dataset.lmdb/` - Base de datos indexada

### Archivo NIfTI Típico:
```
$OASIS3_IMAGE_ROOT/sub-OAS30001ses-d0129_T1w.nii.gz
```

### CSV Típico:
```
../../data/dementia/mni-dataset.csv
```

---

## 🎓 Flujo de Aprendizaje Recomendado

1. **Lectura** (5 min)
   - Lee `QUICK_START.md`

2. **Primer Contacto** (10 min)
   - Ejecuta `python scripts/quick_start.py`
   - Prueba la opción "Analizar dataset"

3. **Exploración** (30 min)
   - Abre el Jupyter notebook
   - Sigue el tutorial paso a paso

4. **Práctica** (variable)
   - Crea tus propias galerías
   - Analiza diferentes volúmenes

5. **Profundización** (variable)
   - Lee `README.md` completo
   - Estudia el código de los scripts
   - Realiza los ejercicios del notebook

---

## ✨ Características Destacadas

- ✅ Visualización en 3 planos ortogonales
- ✅ Visor interactivo con sliders
- ✅ Análisis estadístico automático
- ✅ Creación de galerías visuales
- ✅ Jupyter notebook interactivo
- ✅ Interfaz de menú fácil de usar
- ✅ Documentación completa
- ✅ Ejemplos de código
- ✅ Manejo de formatos NIfTI y LMDB

---

## 🚀 Comienza Aquí

```bash
# Opción 1: Menú interactivo (más fácil)
python scripts/quick_start.py

# Opción 2: Jupyter Notebook (más interactivo)
jupyter notebook notebooks/Explora_Datos_OASIS3.ipynb

# Opción 3: Scripts directos (más control)
python scripts/visualize_mri_volume.py --image_path <ruta>
```

---

¿Preguntas? Revisa `README.md` para más información. 📚
