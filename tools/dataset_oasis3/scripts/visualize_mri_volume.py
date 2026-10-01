"""
Visualizar un volumen MRI completo en 3 planos (axial, sagital, coronal).
Uso: python visualize_mri_volume.py --image_path <ruta_nifti>
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


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


def visualize_volume(volume: np.ndarray, title: str = "MRI Volume", save_path: str | None = None):
    """Visualizar el volumen en los 3 planos principales."""
    x, y, z = volume.shape
    mid_x, mid_y, mid_z = x // 2, y // 2, z // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Plano Axial (vista superior)
    ax_axial = axes[0]
    ax_axial.imshow(volume[:, :, mid_z], cmap='gray')
    ax_axial.set_title(f'Axial (Z={mid_z})')
    ax_axial.set_xlabel('X (Left-Right)')
    ax_axial.set_ylabel('Y (Anterior-Posterior)')
    ax_axial.axhline(mid_y, color='r', linestyle='--', alpha=0.3, linewidth=1)
    ax_axial.axvline(mid_x, color='r', linestyle='--', alpha=0.3, linewidth=1)

    # Plano Sagital (vista lateral)
    ax_sagital = axes[1]
    ax_sagital.imshow(volume[mid_x, :, :], cmap='gray', origin='lower')
    ax_sagital.set_title(f'Sagital (X={mid_x})')
    ax_sagital.set_xlabel('Y (Anterior-Posterior)')
    ax_sagital.set_ylabel('Z (Inferior-Superior)')
    ax_sagital.axhline(mid_z, color='r', linestyle='--', alpha=0.3, linewidth=1)
    ax_sagital.axvline(mid_y, color='r', linestyle='--', alpha=0.3, linewidth=1)

    # Plano Coronal (vista frontal)
    ax_coronal = axes[2]
    ax_coronal.imshow(volume[:, mid_y, :], cmap='gray', origin='lower')
    ax_coronal.set_title(f'Coronal (Y={mid_y})')
    ax_coronal.set_xlabel('X (Left-Right)')
    ax_coronal.set_ylabel('Z (Inferior-Superior)')
    ax_coronal.axhline(mid_z, color='r', linestyle='--', alpha=0.3, linewidth=1)
    ax_coronal.axvline(mid_x, color='r', linestyle='--', alpha=0.3, linewidth=1)

    for ax in axes:
        ax.grid(True, alpha=0.2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Imagen guardada en {save_path}")

    plt.show()


def print_volume_info(volume: np.ndarray, image_path: str):
    """Imprimir información del volumen."""
    img = nib.load(image_path)
    affine = img.affine

    print("\n" + "="*60)
    print("INFORMACIÓN DEL VOLUMEN MRI")
    print("="*60)
    print(f"Forma (voxels):        {volume.shape}")
    print(f"Tipo de dato:          {volume.dtype}")
    print(f"Rango de valores:      [{volume.min():.4f}, {volume.max():.4f}]")
    print(f"Media:                 {volume.mean():.4f}")
    print(f"Desviación estándar:   {volume.std():.4f}")
    print(f"Voxels no-cero:        {np.count_nonzero(volume)} / {volume.size}")
    print(f"Matriz de afinidad:\n{affine}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Visualizar un volumen MRI en 3 planos ortogonales'
    )
    parser.add_argument(
        '--image_path',
        required=True,
        help='Ruta al archivo NIfTI (.nii.gz o .nii)'
    )
    parser.add_argument(
        '--title',
        default='MRI Volume',
        help='Título de la visualización'
    )
    parser.add_argument(
        '--save',
        help='Guardar la imagen en esta ruta (PNG)'
    )

    args = parser.parse_args()
    image_path = Path(args.image_path)

    if not image_path.exists():
        print(f"❌ Error: Archivo no encontrado: {image_path}")
        return

    print(f"📂 Cargando: {image_path}")
    volume = load_and_normalize_volume(image_path)
    print_volume_info(volume, str(image_path))

    visualize_volume(volume, title=args.title, save_path=args.save)


if __name__ == "__main__":
    main()
