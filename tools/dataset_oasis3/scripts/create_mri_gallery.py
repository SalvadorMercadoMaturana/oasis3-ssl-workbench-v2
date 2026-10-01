"""
Crear galerías de rebanadas MRI para visualizar rápidamente múltiples volúmenes.
Extrae rebanadas axiales y crea una composición visual.
Uso: python create_mri_gallery.py --image_path <ruta_nifti> --output <ruta_png>
"""

import argparse
from pathlib import Path

import matplotlib.gridspec as gridspec
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


def extract_slices(volume: np.ndarray, z_positions: list[int]) -> list[np.ndarray]:
    """Extraer rebanadas axiales en posiciones específicas."""
    slices = []
    for z in z_positions:
        if 0 <= z < volume.shape[2]:
            slices.append(volume[:, :, z])
        else:
            # Si la posición está fuera de rango, usar 0
            slices.append(np.zeros_like(volume[:, :, 0]))
    return slices


def create_slice_gallery(volume: np.ndarray, title: str = "", save_path: str | None = None):
    """Crear galería de 9 rebanadas equidistantes."""
    z_size = volume.shape[2]

    # Seleccionar 9 rebanadas equidistantes (excluir bordes)
    z_positions = np.linspace(int(z_size * 0.1), int(z_size * 0.9), 9, dtype=int)
    slices = extract_slices(volume, z_positions)

    # Crear figura en grid 3x3
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle(f'{title}\nRebanadas Axiales (Centro del Volumen)',
                 fontsize=14, fontweight='bold', y=0.98)

    for idx, (z_pos, slice_data) in enumerate(zip(z_positions, slices, strict=True)):
        ax = axes[idx // 3, idx % 3]
        im = ax.imshow(slice_data, cmap='gray')
        ax.set_title(f'Z = {z_pos}', fontsize=10, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.2)

        # Colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Galería guardada en {save_path}")

    plt.show()


def create_advanced_gallery(volume: np.ndarray, title: str = "", save_path: str | None = None):
    """Crear galería avanzada con contexto anatómico."""
    z_size = volume.shape[2]

    # Seleccionar rebanadas con rango más amplio
    z_positions = np.linspace(int(z_size * 0.05), int(z_size * 0.95), 12, dtype=int)
    slices = extract_slices(volume, z_positions)

    # Crear figura con GridSpec para mejor control
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'{title}\nVista Completa del Volumen (Rebanadas Axiales)',
                 fontsize=14, fontweight='bold')

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

    for idx, (z_pos, slice_data) in enumerate(zip(z_positions, slices, strict=True)):
        ax = fig.add_subplot(gs[idx // 4, idx % 4])

        # Descripción anatómica aproximada
        z_fraction = z_pos / z_size
        if z_fraction < 0.3:
            anatomy = "Inferior (Base del cerebro)"
        elif z_fraction < 0.7:
            anatomy = "Central"
        else:
            anatomy = "Superior (Vértex)"

        ax.imshow(slice_data, cmap='gray')
        ax.set_title(f'Z = {z_pos}\n{anatomy}', fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Galería avanzada guardada en {save_path}")

    plt.show()


def create_comparison_slices(volumes: list[tuple[str, np.ndarray]],
                             output_path: str | None = None):
    """Crear comparación lado a lado de múltiples volúmenes."""
    if len(volumes) > 4:
        print("⚠️  Solo se pueden comparar hasta 4 volúmenes")
        volumes = volumes[:4]

    num_volumes = len(volumes)
    z_size = volumes[0][1].shape[2]
    z_positions = np.linspace(int(z_size * 0.2), int(z_size * 0.8), 5, dtype=int)

    fig, axes = plt.subplots(num_volumes, len(z_positions),
                             figsize=(15, 3 * num_volumes))
    fig.suptitle('Comparación de Volúmenes MRI', fontsize=14, fontweight='bold')

    if num_volumes == 1:
        axes = axes.reshape(1, -1)

    for vol_idx, (label, volume) in enumerate(volumes):
        slices = extract_slices(volume, z_positions)
        for slice_idx, (z_pos, slice_data) in enumerate(
            zip(z_positions, slices, strict=True)
        ):
            ax = axes[vol_idx, slice_idx]
            ax.imshow(slice_data, cmap='gray')

            if vol_idx == 0:
                ax.set_title(f'Z = {z_pos}', fontsize=10)
            if slice_idx == 0:
                ax.set_ylabel(label, fontsize=11, fontweight='bold')

            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Comparación guardada en {output_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Crear galerías visuales de volúmenes MRI'
    )
    parser.add_argument(
        '--image_path',
        required=True,
        help='Ruta al archivo NIfTI'
    )
    parser.add_argument(
        '--output',
        help='Ruta de salida para guardar la galería (PNG)'
    )
    parser.add_argument(
        '--style',
        choices=['simple', 'advanced'],
        default='simple',
        help='Estilo de galería: simple (3x3) o advanced (3x4 con anatomía)'
    )
    parser.add_argument(
        '--title',
        default='MRI Gallery',
        help='Título de la galería'
    )

    args = parser.parse_args()
    image_path = Path(args.image_path)

    if not image_path.exists():
        print(f"❌ Error: Archivo no encontrado: {image_path}")
        return

    print(f"📂 Cargando: {image_path}")
    volume = load_and_normalize_volume(image_path)
    print(f"✓ Volumen cargado. Dimensiones: {volume.shape}")

    if args.style == 'simple':
        print("🎨 Creando galería simple (3x3)...")
        create_slice_gallery(volume, title=args.title, save_path=args.output)
    else:
        print("🎨 Creando galería avanzada (3x4)...")
        create_advanced_gallery(volume, title=args.title, save_path=args.output)


if __name__ == "__main__":
    main()
