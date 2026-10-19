# MEG-APSU

**Structure-based triage of hydrogen-tunnelling enzymes.**

Given a PDB structure, MEG-APSU decides in milliseconds whether the active site contains
a catalytically competent centre for hydrogen transfer — radical C–H abstraction or
hydride transfer, the reaction classes in which enzymatic H-tunnelling is observed. The
intended use is to rank enzymes *before* an experiment: which ones are worth a kinetic
isotope effect (KIE) measurement or a QM/MM study with tunnelling.

We are not aware of a published tool for this specific task. The closest related work
classifies metal sites as enzymatic or non-enzymatic with machine learning
(MAHOMES; Feehan, Franklin & Slusky, *Nat. Commun.* 2021, doi:10.1038/s41467-021-24070-3); MEG-APSU's
catalysis filter addresses a similar sub-question with explicit physical rules.

## How it works

1. **Motif rules.** Heme, catalytic metals (Fe, Cu, Mn, Co, Mo), NAD(P)/FAD/FMN,
   quinone cofactors, folates, pterins, SAM, Ser/Cys-His-Asp triads, Zn sites and the
   2-His-1-carboxylate facial triad are located in the structure and assigned a
   reaction class (radical C–H, hydride, proton relay, Lewis acid).
2. **Catalysis filter (v1.3).** A metal or heme centre counts only if its geometry allows
   catalysis: saturated coordination, Fe–S clusters, type-1 Cu, heme with two axial
   protein ligands, or a globin-like heme without a distal push–pull partner are
   classified as electron-transfer or transport sites and excluded.
3. **Score.** An APSU score from 0 to 100 combines the motif counts, substrate presence
   and a class prior. Score ≥ 10 counts as positive.

   | score | class |
   |---|---|
   | ≥ 40 | `TUNNELLING-LIKELY` |
   | 10–40 | `TUNNELLING-POSSIBLE` |
   | < 10 | `TUNNELLING-UNLIKELY` |

**What the score is not.** The class prior comes from a two-level Lindblad model whose
barrier and distance are fixed per motif type; it is a constant per motif type and is
*not* computed from atomic coordinates. MEG-APSU therefore does not predict tunnelling
factors. The KIE output is a per-class order of magnitude (geometric mean of literature
values), not a prediction for the individual enzyme.

## Benchmark

All test sets are split by protein family and were frozen before evaluation. Each model
is compared with a trivial baseline: "does the structure contain any redox cofactor or
redox metal?" A model that does not beat this baseline measures nothing beyond it.

| set | proteins | MEG-APSU accuracy (95 % CI) | baseline | non-catalytic redox controls flagged |
|---|---|---|---|---|
| DEV (development) | 66 | 98.5 % (91.9–99.7) | 86.4 % | 0 / 7 |
| TEST, frozen 2026-10-06 | 40 | **92.5 % (80.1–97.4)** | 75.0 % | 3 / 10 |
| TEST v1.4, frozen 2026-10-14, new families | 21 | **66.7 % (45.4–82.8)** | 52.4 % | 2 / 8 |

On TEST v1.4, sensitivity is **1 of 6**: enzymes with cofactors outside the motif list
(B12, Ni-F430, PQQ) are missed. This is the main known limitation. Because TEST v1.4
has now been seen, any extension of the cofactor list must be evaluated on a new frozen
set.

Reproduce:

```bash
cargo build --release
./target/release/meg-apsu bench test      # frozen 2026-10-06
./target/release/meg-apsu bench test14    # frozen 2026-10-14
./target/release/meg-apsu bench test15    # frozen 2026-10-19 (54 new structures; not yet evaluated)
./target/release/meg-apsu bench dev
```

Structures are downloaded from RCSB on first use and cached in `~/.meg-apsu-pdb-cache`.

## Usage

```bash
./target/release/meg-apsu scan 1YGE.pdb            # one structure
./target/release/meg-apsu scan 1YGE.pdb --json
./target/release/meg-apsu batch ./structures/      # every .pdb in a directory
```

## Data

- `data/tunnelling_signatures.csv` — experimental signatures of H-tunnelling for 55
  enzymes (H/D KIE, A_H/A_D, temperature dependence of the KIE, Swain–Schaad exponent)
  with short references. References are not yet DOI-checked.
- `data/fda_drug_target_enzymes.csv` — 115 enzyme drug targets with PDB entries, for
  batch triage.

## Limitations

- Cofactor coverage: B12, F430, PQQ and other cofactors outside the motif list are not
  recognised.
- Apo structures, or structures without a bound substrate or cofactor, can be missed.
- The score does not rank enzymes by KIE size (TEST v1.4: Spearman ρ = −0.35, n = 5).
- Small test sets: confidence intervals are wide.

## Changes in v1.5

Labels and commands only; every score is bit-identical to v1.4 (checked on 127
structures and all three benchmarks).

- New description; classes renamed to `TUNNELLING-LIKELY / POSSIBLE / UNLIKELY`, aligned
  with the benchmark threshold.
- The Lindblad output is reported as a class prior and documented as structure-independent.
- Removed the commands `validate` and `blind` (superseded by `bench`), `proof` and
  `drugbank`, together with `how-it-works.html`. They made claims about classical
  mechanics, docking errors and drug withdrawals that the method does not support.
  Their literature data are kept in `data/`. Everything remains in the git history.

## About

Written and maintained by Sabrina Hackfort (sectio-aurea-q, MEGALODON Research),
ORCID [0009-0007-7458-2487](https://orcid.org/0009-0007-7458-2487), as an independent
researcher. Built on a self-repaired MacBook.

## License

AGPL-3.0-or-later (from v1.5.1). Versions up to v1.5 were released under MIT.
If you use MEG-APSU in research, please cite it (see CITATION.cff).
