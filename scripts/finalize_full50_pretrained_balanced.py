"""Evaluate the 50-epoch pretrained campaign retrained on the balanced split.

Same training as ``campaign_full50_pretrained`` (see
``finalize_full50_pretrained.py``); only the manifest changes to
``train_sessions_eval_primary_balanced.csv``, a subset of the ``campaign_full50``
split in which every kept subject stays in its original split. Because the
balanced test subjects are a subset of the original test subjects, the report
also rescores the original campaign's probe predictions on exactly those
subjects, so both campaigns are compared on identical test subjects.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from evaluate_frozen_probe import paired_auc_difference
from finalize_full50_pretrained import METHOD_LABEL, ROOT, experiment_paths, main
from sklearn.metrics import roc_auc_score

CAMPAIGN = ROOT / "campaign_full50_pretrained_balanced"
MANIFEST = CAMPAIGN / "train_sessions_eval_primary_balanced.csv"
ORIGINAL = ROOT / "campaign_full50_pretrained"
ORIGINAL_MANIFEST = ROOT / "campaign_full50" / "train_sessions_eval_primary.csv"
SPLITS = ("train", "val", "test")


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def split_summary() -> dict:
    """Describe the balanced manifest relative to the campaign_full50 split."""
    new, old = read_manifest(MANIFEST), read_manifest(ORIGINAL_MANIFEST)
    old_split = {row["subject"]: row["split"] for row in old}
    old_group = {row["subject"]: row["group"] for row in old if row["primary"] == "1"}
    summary = {"subjects": len({row["subject"] for row in new}), "splits": {}}
    for split in SPLITS:
        new_subjects = {row["subject"] for row in new if row["split"] == split}
        old_subjects = {row["subject"] for row in old if row["split"] == split}
        if any(old_split.get(subject) != split for subject in new_subjects):
            raise RuntimeError(f"Balanced {split} subjects are not a subset of the original {split}")
        summary["splits"][split] = {
            "subjects": len(new_subjects),
            "sessions": sum(row["split"] == split for row in new),
            "original_subjects": len(old_subjects),
            "original_sessions": sum(row["split"] == split for row in old),
            "dropped_groups": dict(Counter(old_group[s] for s in old_subjects - new_subjects)),
        }
    primary = {row["subject"] for row in new if row["primary"] == "1"}
    summary["train_subjects_without_primary"] = len(
        {row["subject"] for row in new if row["split"] == "train"} - primary)
    return summary


def describe_split(summary: dict) -> str:
    parts = []
    for split in SPLITS:
        info = summary["splits"][split]
        dropped = ", ".join(f"{n} {group}" for group, n in
                            sorted(info["dropped_groups"].items(), key=lambda kv: -kv[1]))
        parts.append(f"{split}, {info['subjects']} de {info['original_subjects']} sujetos "
                     f"(se quitaron {dropped or 'ninguno'})")
    return "; ".join(parts)


def comparison_lines(rows: list[dict]) -> list[str]:
    """Rescore the original campaign's probes on the balanced test subjects."""
    original_paths = experiment_paths(ORIGINAL)
    table = []
    for row in rows:
        method = row["method"]
        new_doc = json.loads((ROOT / row["experiment_dir"] /
                              "ad_hc_subject_probe_best.json").read_text(encoding="utf-8"))
        old_doc = json.loads((original_paths[method] /
                              "ad_hc_subject_probe_best.json").read_text(encoding="utf-8"))
        for kind in ("trained", "random"):
            new_result = new_doc["results"][kind]
            old_result = old_doc["results"][kind]
            new_pred = new_result["test_predictions"]
            old_by_subject = {item["subject"]: item for item in old_result["test_predictions"]}
            missing = [item["subject"] for item in new_pred if item["subject"] not in old_by_subject]
            if missing:
                raise RuntimeError(f"{method}: balanced test subjects absent from the original "
                                   f"test: {missing[:5]}")
            same = [old_by_subject[item["subject"]] for item in new_pred]
            if [item["AD"] for item in same] != [item["AD"] for item in new_pred]:
                raise RuntimeError(f"{method}: AD/HC labels differ between campaigns")
            labels = [item["AD"] for item in same]
            old_same_auc = float(roc_auc_score(labels, [item["score"] for item in same]))
            comparison = paired_auc_difference(
                {"test_predictions": new_pred, "test_auc": new_result["test_auc"]},
                {"test_predictions": same, "test_auc": old_same_auc})
            table.append({
                "method": method, "kind": kind,
                "original_full_auc": old_result["test_auc"],
                "original_full_n": old_result["test_subjects"],
                "original_same_auc": old_same_auc,
                "balanced_auc": new_result["test_auc"], "n": len(new_pred),
                "original_train_n": old_result["train_subjects"],
                "balanced_train_n": new_result["train_subjects"],
                "delta": comparison["delta_test_auc"],
                "ci": comparison["delta_test_auc_bootstrap_95ci"],
                "p_positive": comparison["bootstrap_probability_delta_positive"],
            })
    (CAMPAIGN / "comparison_vs_original.json").write_text(json.dumps(table, indent=2) + "\n")

    first = table[0]
    lines = ["## Comparación con `campaign_full50_pretrained` (mismos sujetos de test)", "",
             f"El test balanceado es un subconjunto del test original, así que las predicciones "
             f"del probe de la campaña original se vuelven a puntuar solo sobre los mismos "
             f"{first['n']} sujetos AD/HC. Cada probe se ajustó con el train de su propia "
             f"campaña ({first['original_train_n']} sujetos AD/HC en la original, "
             f"{first['balanced_train_n']} en la balanceada), así que Δ mezcla el efecto del "
             f"encoder SSL y el del conjunto con que se ajustó el probe. F1-macro y AUPR no se "
             f"comparan porque dependen de la prevalencia de AD, que cambia entre ambos tests.", "",
             f"| Método | Encoder | AUROC original, test completo (n={first['original_full_n']}) | "
             f"AUROC original, mismos sujetos (n={first['n']}) | "
             f"AUROC balanceado (n={first['n']}) | Δ balanceado − original [IC 95 %] | P(Δ>0) |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for item in table:
        lo, hi = item["ci"]
        lines.append(
            f"| {METHOD_LABEL[item['method']]} | "
            f"{'SSL' if item['kind'] == 'trained' else 'Aleatorio'} | "
            f"{item['original_full_auc']:.3f} | {item['original_same_auc']:.3f} | "
            f"{item['balanced_auc']:.3f} | {item['delta']:+.3f} [{lo:+.3f}, {hi:+.3f}] | "
            f"{item['p_positive']:.3f} |")
    lines += ["", "IC 95 %: bootstrap pareado por sujeto dentro de clase sobre los sujetos de "
              "test comunes. P(Δ>0) cercano a 0,5 indica que no hay diferencia distinguible.", ""]
    return lines


if __name__ == "__main__":
    info = split_summary()
    splits = info["splits"]
    main(
        CAMPAIGN,
        report_title="# OASIS3: SSL con backbones pre-entrenados en ImageNet, 50 épocas, "
              "split balanceado AD/HC",
        intro=(
            "Mismo entrenamiento que [`campaign_full50_pretrained`]"
            "(../campaign_full50_pretrained/RESULTADOS.md): mismos modelos e "
            "hiperparámetros (VICReg con ConvNeXt-tiny ImageNet1K y dropout=0.1 en el "
            "proyector, BYOL con ResNet-18 ImageNet1K, DINOv2 con ViT-Small/patch60 desde "
            "cero), 50 épocas sin early stopping, entrenados uno a la vez en GPU (NVIDIA "
            "GTX 1650) a precisión completa (32-true) con `trainer.deterministic=false`, así "
            "que no son reproducibles bit a bit. Solo cambia el split: el manifest "
            "`train_sessions_eval_primary_balanced.csv`, con "
            f"{info['subjects']} sujetos OASIS3. Es un subconjunto del split de "
            "`campaign_full50`: cada sujeto conserva su split original y no hay sujetos "
            f"compartidos entre splits. Por split: {describe_split(info)}. En train quedan "
            f"{splits['train']['sessions']} sesiones (de "
            f"{splits['train']['original_sessions']}). SSL sigue sorteando una sesión y un "
            "corte por sujeto de train en cada época; validación y test usan una sesión y un "
            f"corte fijos. {info['train_subjects_without_primary']} sujetos de train no "
            "tienen ninguna sesión marcada como primaria en el manifest; como la campaña usa "
            "`primary_only=false`, entrenan y entran al probe con las sesiones que quedaron. "
            "Los checkpoints se guardaron sin estado del optimizador "
            "(`trainer.checkpoint_weights_only=true`), lo que no afecta al entrenamiento."),
        protocol=(
            "Same training as campaign_full50_pretrained (models, hyperparameters, fixed 50 "
            "epochs without early stopping, sequential GPU runs at 32-true with "
            "trainer.deterministic=false); only the manifest changes to "
            f"train_sessions_eval_primary_balanced.csv: {info['subjects']} subjects, a "
            "subject-level subset of the campaign_full50 split (every kept subject keeps its "
            f"original split) -- {json.dumps(splits)}; "
            f"{info['train_subjects_without_primary']} train subjects have no primary "
            "session and use their remaining sessions (primary_only=false). SSL sees one "
            "randomly selected session and axial slice per train subject per epoch; fixed "
            "session and central slice for validation and frozen linear probe; checkpoint "
            "chosen without labels (best val/loss SSL checkpoint); probe C chosen on "
            "validation AUROC; test evaluated once. Checkpoints saved weights-only."),
        script_name="finalize_full50_pretrained_balanced.py",
        extra_lines=comparison_lines,
    )
