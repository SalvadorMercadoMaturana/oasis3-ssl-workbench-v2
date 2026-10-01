"""
Visor interactivo de rebanadas MRI con controles deslizantes.
Usa matplotlib para navegación interactiva por los 3 planos.
Uso: python interactive_viewer.py --image_path <ruta_nifti>
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.widgets import Slider


def load_and_normalize_volume(image_path: str | Path) -> np.ndarray:
    """Cargar NIfTI y normalizar a [0, 1]."""
    img = nib.load(image_path)
    volume = np.asarray(nib.as_closest_canonical(img).get_fdata(dtype=np.float32))

    # Normalización robusta
    foreground = volume[volume != 0]
    if foreground.size > 0:
        low, high = np.percentile(foreground, [1, 99])
        volume = np.clip((volume - low) / max(high - low, 1e-6), 0, 1)

    return volume


class MRIViewer:
    def __init__(self, volume: np.ndarray, title: str = "MRI Viewer"):
        self.volume = volume
        self.title = title
        self.x_size, self.y_size, self.z_size = volume.shape

        # Slices iniciales
        self.x_slice = self.x_size // 2
        self.y_slice = self.y_size // 2
        self.z_slice = self.z_size // 2

        self.setup_plot()

    def setup_plot(self):
        """Crear figura con tres subplots y sliders."""
        self.fig = plt.figure(figsize=(14, 10))
        self.fig.suptitle(self.title, fontsize=14, fontweight='bold')

        # Subplots para los 3 planos
        self.ax_axial = plt.subplot(2, 2, 1)
        self.ax_sagital = plt.subplot(2, 2, 2)
        self.ax_coronal = plt.subplot(2, 2, 3)
        ax_info = plt.subplot(2, 2, 4)
        ax_info.axis('off')

        self.info_text = ax_info.text(
            0.1, 0.5, '', fontsize=10, family='monospace',
            verticalalignment='center'
        )

        # Espacios para sliders
        plt.subplots_adjust(left=0.2, bottom=0.35, right=0.9, top=0.95)

        # Sliders
        ax_slider_x = plt.axes([0.2, 0.25, 0.7, 0.02])
        ax_slider_y = plt.axes([0.2, 0.20, 0.7, 0.02])
        ax_slider_z = plt.axes([0.2, 0.15, 0.7, 0.02])

        self.slider_x = Slider(
            ax_slider_x, 'Sagital (X)', 0, self.x_size - 1,
            valinit=self.x_slice, valstep=1, color='red'
        )
        self.slider_y = Slider(
            ax_slider_y, 'Coronal (Y)', 0, self.y_size - 1,
            valinit=self.y_slice, valstep=1, color='green'
        )
        self.slider_z = Slider(
            ax_slider_z, 'Axial (Z)', 0, self.z_size - 1,
            valinit=self.z_slice, valstep=1, color='blue'
        )

        self.slider_x.on_changed(self.update)
        self.slider_y.on_changed(self.update)
        self.slider_z.on_changed(self.update)

        self.update()

    def update(self, val=None):
        """Actualizar visualización cuando se mueve un slider."""
        self.x_slice = int(self.slider_x.val)
        self.y_slice = int(self.slider_y.val)
        self.z_slice = int(self.slider_z.val)

        # Limpiar subplots
        self.ax_axial.clear()
        self.ax_sagital.clear()
        self.ax_coronal.clear()

        # Plano Axial (Z)
        self.ax_axial.imshow(self.volume[:, :, self.z_slice], cmap='gray')
        self.ax_axial.axhline(self.y_slice, color='g', linestyle='--', alpha=0.5, linewidth=1)
        self.ax_axial.axvline(self.x_slice, color='r', linestyle='--', alpha=0.5, linewidth=1)
        self.ax_axial.set_title(f'Axial (Z={self.z_slice})', fontsize=11, fontweight='bold')
        self.ax_axial.set_xlabel('X')
        self.ax_axial.set_ylabel('Y')
        self.ax_axial.grid(True, alpha=0.2)

        # Plano Sagital (X)
        self.ax_sagital.imshow(
            self.volume[self.x_slice, :, :], cmap='gray', origin='lower'
        )
        self.ax_sagital.axhline(self.z_slice, color='b', linestyle='--', alpha=0.5, linewidth=1)
        self.ax_sagital.axvline(self.y_slice, color='g', linestyle='--', alpha=0.5, linewidth=1)
        self.ax_sagital.set_title(f'Sagital (X={self.x_slice})', fontsize=11, fontweight='bold')
        self.ax_sagital.set_xlabel('Y')
        self.ax_sagital.set_ylabel('Z')
        self.ax_sagital.grid(True, alpha=0.2)

        # Plano Coronal (Y)
        self.ax_coronal.imshow(
            self.volume[:, self.y_slice, :], cmap='gray', origin='lower'
        )
        self.ax_coronal.axhline(self.z_slice, color='b', linestyle='--', alpha=0.5, linewidth=1)
        self.ax_coronal.axvline(self.x_slice, color='r', linestyle='--', alpha=0.5, linewidth=1)
        self.ax_coronal.set_title(f'Coronal (Y={self.y_slice})', fontsize=11, fontweight='bold')
        self.ax_coronal.set_xlabel('X')
        self.ax_coronal.set_ylabel('Z')
        self.ax_coronal.grid(True, alpha=0.2)

        # Actualizar información
        voxel_value = self.volume[self.x_slice, self.y_slice, self.z_slice]
        info = f"""POSICIÓN ACTUAL

Voxel: ({self.x_slice}, {self.y_slice}, {self.z_slice})
Valor: {voxel_value:.4f}

VOLUMEN COMPLETO
Forma: {self.volume.shape}
Min: {self.volume.min():.4f}
Max: {self.volume.max():.4f}
Media: {self.volume.mean():.4f}
Std: {self.volume.std():.4f}

ATAJOS:
Q/Esc: Salir
"""
        self.info_text.set_text(info)

        self.fig.canvas.draw_idle()

    def show(self):
        """Mostrar el visor."""
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Visor interactivo de volúmenes MRI con sliders'
    )
    parser.add_argument(
        '--image_path',
        required=True,
        help='Ruta al archivo NIfTI'
    )
    parser.add_argument(
        '--title',
        default='MRI Viewer Interactivo',
        help='Título de la ventana'
    )

    args = parser.parse_args()
    image_path = Path(args.image_path)

    if not image_path.exists():
        print(f"❌ Error: Archivo no encontrado: {image_path}")
        return

    print(f"📂 Cargando: {image_path}")
    volume = load_and_normalize_volume(image_path)
    print("✓ Volumen cargado")
    print(f"  Dimensiones: {volume.shape}")
    print("\n💡 Usa los sliders para navegar por los 3 planos ortogonales")

    viewer = MRIViewer(volume, title=args.title)
    viewer.show()


if __name__ == "__main__":
    main()
