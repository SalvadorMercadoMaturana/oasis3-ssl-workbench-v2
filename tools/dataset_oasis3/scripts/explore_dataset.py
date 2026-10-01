"""
Explorar estadísticas del dataset OASIS3.
Lee el CSV clínico y muestra distribuciones de grupos.
Uso: python explore_dataset.py --csv_path <ruta_csv>
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

SUBJECT_RE = re.compile(r"^(sub-OAS\d+)ses")


def subject_from_record(record: dict) -> str:
    """Return the explicit subject or derive it from an OASIS3 session code."""
    subject = record.get('subject', '').strip()
    if subject:
        return subject
    code = record.get('original_code', '').strip()
    match = SUBJECT_RE.match(code)
    if match is None:
        raise ValueError(f"No se puede derivar el sujeto desde original_code={code!r}")
    return match.group(1)


def read_oasis3_csv(csv_path: str | Path) -> list[dict]:
    """Leer CSV y filtrar solo registros OASIS3."""
    records = []
    with open(csv_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('study_name', '').strip().upper() == 'OASIS3':
                records.append(row)
    return records


def analyze_dataset(records: list[dict]) -> dict:
    """Analizar estadísticas del dataset."""
    stats = {
        'total_records': len(records),
        'groups': defaultdict(int),
        'records_per_subject': defaultdict(int),
        'sessions_per_subject': defaultdict(set),
    }

    for record in records:
        # Extraer información
        subject = subject_from_record(record)
        group = record.get('group', 'unknown').strip().upper()
        session = record.get('original_code', 'unknown')

        # Contar grupos
        stats['groups'][group] += 1

        # Contar registros por sujeto
        stats['records_per_subject'][subject] += 1

        # Sesiones por sujeto
        stats['sessions_per_subject'][subject].add(session)

    # Convertir sets a conteos
    stats['unique_sessions_per_subject'] = {
        subject: len(sessions)
        for subject, sessions in stats['sessions_per_subject'].items()
    }
    stats['total_subjects'] = len(stats['records_per_subject'])

    return stats


def print_stats(stats: dict):
    """Imprimir estadísticas formateadas."""
    print("\n" + "="*70)
    print("ESTADÍSTICAS DEL DATASET OASIS3")
    print("="*70)

    print("\n📊 RESUMEN GENERAL")
    print(f"  Total de registros (records):  {stats['total_records']:,}")
    print(f"  Total de sujetos únicos:       {stats['total_subjects']:,}")

    # Promedio de registros por sujeto
    records_per_subj = list(stats['records_per_subject'].values())
    print(f"  Registros por sujeto (media):  {np.mean(records_per_subj):.2f}")
    print(f"  Registros por sujeto (rango):  {min(records_per_subj)} - {max(records_per_subj)}")

    # Promedio de sesiones por sujeto
    sessions_per_subj = list(stats['unique_sessions_per_subject'].values())
    print(f"  Sesiones por sujeto (media):   {np.mean(sessions_per_subj):.2f}")
    print(f"  Sesiones por sujeto (máx):     {max(sessions_per_subj)}")

    print("\n👥 DISTRIBUCIÓN POR GRUPO CLÍNICO")
    for group, count in sorted(stats['groups'].items(), key=lambda x: x[1], reverse=True):
        percentage = 100 * count / stats['total_records']
        print(f"  {group:15s}: {count:6,} ({percentage:5.1f}%)")

    print("\n" + "="*70 + "\n")


def visualize_stats(records: list[dict], stats: dict, save_path: str | None = None):
    """Crear visualizaciones de estadísticas."""
    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    fig.suptitle('OASIS3 Dataset - Análisis Estadístico', fontsize=16, fontweight='bold')

    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Distribución de grupos
    ax1 = fig.add_subplot(gs[0, 0])
    groups = sorted(stats['groups'].items(), key=lambda x: x[1], reverse=True)
    group_names = [g[0] for g in groups]
    group_counts = [g[1] for g in groups]
    colors = plt.cm.Set3(np.linspace(0, 1, len(group_names)))

    bars = ax1.bar(group_names, group_counts, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_title('Distribución de Grupos Clínicos', fontweight='bold')
    ax1.set_ylabel('Número de Registros')
    ax1.grid(axis='y', alpha=0.3)

    # Etiquetas en barras
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')

    # 2. Registros por sujeto (histograma)
    ax2 = fig.add_subplot(gs[0, 1])
    records_per_subj = list(stats['records_per_subject'].values())
    ax2.hist(records_per_subj, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax2.set_title('Distribución: Registros por Sujeto', fontweight='bold')
    ax2.set_xlabel('Número de Registros')
    ax2.set_ylabel('Número de Sujetos')
    ax2.axvline(np.mean(records_per_subj), color='red', linestyle='--',
                linewidth=2, label=f'Media: {np.mean(records_per_subj):.1f}')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 3. Sesiones por sujeto (histograma)
    ax3 = fig.add_subplot(gs[1, 0])
    sessions_per_subj = list(stats['unique_sessions_per_subject'].values())
    ax3.hist(sessions_per_subj, bins=range(1, max(sessions_per_subj) + 2),
            color='lightcoral', edgecolor='black', alpha=0.7)
    ax3.set_title('Distribución: Sesiones por Sujeto', fontweight='bold')
    ax3.set_xlabel('Número de Sesiones')
    ax3.set_ylabel('Número de Sujetos')
    ax3.axvline(np.mean(sessions_per_subj), color='blue', linestyle='--',
                linewidth=2, label=f'Media: {np.mean(sessions_per_subj):.1f}')
    ax3.legend()
    ax3.grid(alpha=0.3)

    # 4. Información estadística (texto)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    info_text = f"""ESTADÍSTICAS RESUMIDAS

Total Registros:           {stats['total_records']:,}
Total Sujetos:             {stats['total_subjects']:,}

Registros por Sujeto:
  · Media:                 {np.mean(records_per_subj):.2f}
  · Mediana:               {np.median(records_per_subj):.0f}
  · Rango:                 {min(records_per_subj)}-{max(records_per_subj)}

Sesiones por Sujeto:
  · Media:                 {np.mean(sessions_per_subj):.2f}
  · Mediana:               {np.median(sessions_per_subj):.0f}
  · Máximo:                {max(sessions_per_subj)}

Grupos Representados:
"""
    for group, count in sorted(stats['groups'].items(), key=lambda x: x[1], reverse=True):
        pct = 100 * count / stats['total_records']
        info_text += f"\n  · {group:15s}: {count:6,} ({pct:5.1f}%)"

    ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox={'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5})

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Gráfico guardado en {save_path}")
        plt.close(fig)
    else:
        plt.show()


def save_stats_json(stats: dict, output_path: str):
    """Guardar estadísticas en JSON."""
    # Convertir defaultdict a dict regular
    stats_to_save = {
        'total_records': stats['total_records'],
        'total_subjects': stats['total_subjects'],
        'groups': dict(stats['groups']),
        'records_per_subject_stats': {
            'mean': float(np.mean(list(stats['records_per_subject'].values()))),
            'median': float(np.median(list(stats['records_per_subject'].values()))),
            'min': int(min(stats['records_per_subject'].values())),
            'max': int(max(stats['records_per_subject'].values())),
        },
        'sessions_per_subject_stats': {
            'mean': float(np.mean(list(stats['unique_sessions_per_subject'].values()))),
            'max': int(max(stats['unique_sessions_per_subject'].values())),
        }
    }

    with open(output_path, 'w') as f:
        json.dump(stats_to_save, f, indent=2)

    print(f"✓ Estadísticas guardadas en {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Explorar estadísticas del dataset OASIS3'
    )
    parser.add_argument(
        '--csv_path',
        required=True,
        help='Ruta al archivo CSV clínico'
    )
    parser.add_argument(
        '--save_plot',
        help='Guardar gráficos en esta ruta (PNG)'
    )
    parser.add_argument(
        '--save_stats',
        help='Guardar estadísticas en JSON'
    )

    args = parser.parse_args()
    csv_path = Path(args.csv_path)

    if not csv_path.exists():
        print(f"❌ Error: Archivo no encontrado: {csv_path}")
        return

    print(f"📂 Leyendo CSV: {csv_path}")
    records = read_oasis3_csv(csv_path)

    if not records:
        print("❌ No se encontraron registros OASIS3 en el CSV")
        return

    print(f"✓ {len(records)} registros OASIS3 encontrados")

    stats = analyze_dataset(records)
    print_stats(stats)

    if args.save_plot:
        visualize_stats(records, stats, save_path=args.save_plot)
    else:
        visualize_stats(records, stats)

    if args.save_stats:
        save_stats_json(stats, args.save_stats)


if __name__ == "__main__":
    main()
