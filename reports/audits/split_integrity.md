# Split Integrity Audit

Generated: `2026-04-24T08:01:48.075666+00:00`

| Dataset | Participants | Train | Valid | Test | Overlap | Small-dataset CV |
|---|---:|---:|---:|---:|---|---|
| studentlife | 38 | 23 | 8 | 7 | False | 5-fold |
| deprest_cat | 360 | 216 | 72 | 72 | False | none |
| psyche_d | 4948 | 2969 | 990 | 989 | False | none |
| depresjon | 55 | 33 | 11 | 11 | False | 5-fold |
| obf | 159 | 95 | 31 | 33 | False | none |

All splits are participant-level. No anchor-level randomization was used.

Small datasets `studentlife` and `depresjon` also receive a stratified 5-fold subject-level CV manifest.
