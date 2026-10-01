#!/usr/bin/env python3
"""
Script de inicio rápido para Dataset-OASIS3.
Proporciona un menú interactivo para explorar los datos.
"""

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_command(arguments):
    """Run a project command with the active interpreter and repository root."""
    return subprocess.run(arguments, cwd=ROOT, check=False)


def print_banner():
    """Mostrar banner de bienvenida."""
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║                  Dataset-OASIS3 🧠                            ║
    ║        Herramientas de Visualización de Imágenes MRI          ║
    ╚════════════════════════════════════════════════════════════════╝
    """)


def print_menu():
    """Mostrar menú principal."""
    print("\n📋 MENÚ PRINCIPAL")
    print("="*50)
    print("1. Visualizar un volumen MRI (3 planos)")
    print("2. Explorador interactivo (con sliders)")
    print("3. Analizar estadísticas del dataset")
    print("4. Crear galería visual de rebanadas")
    print("5. Abrir Jupyter Notebook")
    print("6. Ver documentación (README)")
    print("0. Salir")
    print("="*50)


def option_visualize():
    """Opción 1: Visualizar volumen."""
    print("\n📂 Ingresa la ruta del archivo NIfTI:")
    image_path = input("Ruta: ").strip()

    if not Path(image_path).exists():
        print("❌ Archivo no encontrado")
        return

    title = input("Título (opcional, Enter para omitir): ").strip() or "MRI Volume"
    save = input("¿Guardar imagen? (s/n): ").strip().lower() == 's'

    cmd = [
        sys.executable, str(ROOT / "scripts/visualize_mri_volume.py"),
        "--image_path", image_path,
        "--title", title
    ]

    if save:
        output_path = input("Ruta de salida (PNG): ").strip()
        cmd.extend(["--save", output_path])

    run_command(cmd)


def option_interactive():
    """Opción 2: Visor interactivo."""
    print("\n🎯 Visor Interactivo")
    print("Ingresa la ruta del archivo NIfTI:")
    image_path = input("Ruta: ").strip()

    if not Path(image_path).exists():
        print("❌ Archivo no encontrado")
        return

    title = input("Título (opcional): ").strip() or "MRI Viewer"

    run_command([
        sys.executable, str(ROOT / "scripts/interactive_viewer.py"),
        "--image_path", image_path,
        "--title", title
    ])


def option_explore():
    """Opción 3: Analizar dataset."""
    print("\n📊 Análisis de Dataset")
    print("Ingresa la ruta del CSV clínico:")
    csv_path = input("Ruta: ").strip()

    if not Path(csv_path).exists():
        print("❌ Archivo no encontrado")
        return

    save_plot = input("¿Guardar gráficos? (s/n): ").strip().lower() == 's'

    cmd = [
        sys.executable, str(ROOT / "scripts/explore_dataset.py"),
        "--csv_path", csv_path
    ]

    if save_plot:
        output = input("Ruta de salida (PNG): ").strip()
        cmd.extend(["--save_plot", output])

    run_command(cmd)


def option_gallery():
    """Opción 4: Crear galería."""
    print("\n🎨 Crear Galería de Rebanadas")
    print("Ingresa la ruta del archivo NIfTI:")
    image_path = input("Ruta: ").strip()

    if not Path(image_path).exists():
        print("❌ Archivo no encontrado")
        return

    output_path = input("Ruta de salida (PNG): ").strip()

    print("\nEstilo de galería:")
    print("1. Simple (3×3 rebanadas)")
    print("2. Avanzado (3×4 con descripción anatómica)")
    style_choice = input("Elige (1/2): ").strip()
    style = "advanced" if style_choice == "2" else "simple"

    title = input("Título (opcional): ").strip() or "MRI Gallery"

    run_command([
        sys.executable, str(ROOT / "scripts/create_mri_gallery.py"),
        "--image_path", image_path,
        "--output", output_path,
        "--style", style,
        "--title", title
    ])


def option_notebook():
    """Opción 5: Abrir Jupyter Notebook."""
    print("\n📔 Abriendo Jupyter Notebook...")
    print("Si no se abre automáticamente, ejecuta:")
    print("  jupyter notebook notebooks/Explora_Datos_OASIS3.ipynb")

    run_command([
        sys.executable, "-m", "jupyter", "notebook",
        str(ROOT / "notebooks/Explora_Datos_OASIS3.ipynb"),
    ])


def option_readme():
    """Opción 6: Ver README."""
    print("\n📖 README.md")
    print("="*70)
    try:
        with (ROOT / "README.md").open("r", encoding="utf-8") as f:
            print(f.read())
    except FileNotFoundError:
        print("❌ README.md no encontrado")
    print("="*70)


def main(argv=None):
    """Loop principal."""
    parser = argparse.ArgumentParser(description="Menú local de herramientas Dataset-OASIS3")
    parser.add_argument(
        "--check",
        action="store_true",
        help="comprobar dependencias y salir sin abrir el menú interactivo",
    )
    args = parser.parse_args(argv)

    print_banner()

    # Verificar dependencias
    print("\n✓ Verificando dependencias...")
    required = ("nibabel", "numpy", "matplotlib", "lmdb")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        print(f"❌ Faltan dependencias: {', '.join(missing)}")
        print("   Ejecuta: pip install -r requirements.txt")
        return
    print("✓ Todas las dependencias están instaladas")
    if args.check:
        return

    while True:
        print_menu()
        try:
            choice = input("\nElige una opción (0-6): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Cancelado")
            break

        try:
            if choice == "0":
                print("\n👋 ¡Hasta luego!")
                break
            elif choice == "1":
                option_visualize()
            elif choice == "2":
                option_interactive()
            elif choice == "3":
                option_explore()
            elif choice == "4":
                option_gallery()
            elif choice == "5":
                option_notebook()
            elif choice == "6":
                option_readme()
            else:
                print("❌ Opción no válida")
        except KeyboardInterrupt:
            print("\n\n👋 Cancelado")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

        try:
            input("\nPresiona Enter para continuar...")
        except EOFError:
            break


if __name__ == "__main__":
    main()
