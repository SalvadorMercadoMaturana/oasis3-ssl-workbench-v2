"""Frozen-encoder evaluation: embeddings, linear probes, metrics, identity checks.

Import the submodules directly. This package imports nothing, so the torch-free ones
(``metrics``, ``protocol_v2``, ``probe_v1``, ``embeddings``, ``curves`` and ``input_qc``)
stay light:

- ``embeddings``: one vector per subject (mean over sessions) of a frozen encoder.
- ``metrics``: AUROC / AUPR / F1-macro / mean3 and threshold metrics.
- ``probe_v1``: the campaign protocol (5x5 CV + train->val->test probe) = result.json cv/test.
- ``protocol_v2``: the protocol since 2026-11-07 (representative CV, log-loss C, auc_age).
- ``identity``: subject identity and collapse under the augmentations.
- ``run_scoring``: what ``train.py`` runs after training (v1 + v2 + spatial probe).
- ``curves``: per-epoch training curves from metrics.csv.
- ``input_qc``: intensity range check of input scans (deployed model, external sets).
"""
