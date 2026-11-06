# test15: evaluation protocol

Fixed on 6 November 2026, before the single evaluation. This file, the code and the
labels are tagged `test15-vorab` and pushed to GitHub before the run, so the timestamp
shows that nothing below was chosen after seeing a test15 result.

## What is evaluated

- **Code:** MEG-APSU v1.6.0 at tag `test15-vorab`, model "v1.6" in `meg-apsu bench test15`.
  Fixed threshold QVS ≥ 10, unchanged since v1.3.
- **Test set:** test15, frozen 19 October 2026 (`test15_set()` in `src/bench.rs`, sources in
  `data/test15_verified.json`): 54 structures, 6 tunnelling, 29 classical, 19 controls.
  No model version has been run on it.
- **Primary analysis:** proteins with strong labels ("nur starke Labels"). Uncertain labels
  are reported but do not count.

## Comparators

Both are fixed in `src/bench.rs` at this tag.

1. **Cofactor rule.** "Does the structure contain a redox cofactor or a redox metal?", with
   the same extended cofactor list as v1.6 (cobalamin, F430 and PQQ added).
2. **EC rule.** Uses the annotation only. Predicts tunnelling if an EC number in the PDB
   header (`COMPND … EC:`) belongs to a class whose reaction cleaves a C–H bond as hydride,
   H atom or proton in the redox step: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.13.11, 1.14, 1.17,
   5.4.99. No EC number, or any other class, means no tunnelling.

## Statistics

- Accuracy, sensitivity and specificity, each with a 95 % Wilson interval.
- Paired comparison of v1.6 with each comparator: exact two-sided McNemar test on the
  discordant proteins, α = 0.05. Two comparisons are pre-specified, and both p values are
  reported without correction.
- Secondary and descriptive only: Spearman ρ between predicted and literature KIE for
  tunnelling proteins with a KIE value.

## What will be concluded

- **v1.6 against the cofactor rule.** If p < 0.05 and v1.6 is better: on test15 the structure
  analysis adds information beyond cofactor presence. Otherwise: on test15 v1.6 cannot be
  distinguished from cofactor presence.
- **v1.6 against the EC rule.** If v1.6 is not significantly better, we state that for
  annotated enzymes the EC number carries the same information, and that the value of
  MEG-APSU then has to come from unannotated structures and from predicting the size of
  the isotope effect. If v1.6 is significantly worse, we say so.
- After the run, code, labels and threshold stay as they are. Any deviation from this
  protocol is reported as a deviation.

## Changes in v1.6

All made and checked on DEV and test14 only.

- **Cobalamin** counts as an H-abstracting cofactor only with a 5'-deoxyadenosyl group at
  the cobalt, or with a substrate within 10 Å of the cobalt (reach of the 5'-deoxyadenosyl
  radical after Co–C cleavage: Co···C5' up to about 6 Å, plus 3–4 Å for the H transfer).
  Without either it is cargo. Methylcobalamin is excluded, because it transfers a methyl
  group.
- **F430 (Ni)** gives a radical C–H site. **PQQ** gives a hydride-transfer site.
- **SAM** counts as H transfer only as radical SAM, that is SAM at a [4Fe-4S] cluster.
  Methyltransferases transfer a methyl group, not hydrogen. Before v1.6 every SAM site was
  scored as hydride transfer.
- **Labels, corrected on 6 November 2026:**
  - COMT (3BWM) is now classical: methyl transfer (SN2), no H transfer in the chemical
    step. The former KIE of 4.5 had no source.
  - IDO (2D0T) is now classical with uncertain evidence: O2 adds to C2=C3 of the indole,
    and no C–H bond is cleaved in the rate-limiting step. The former KIE of 6.0 had no
    source. It did not count in the strong-label results before and does not count now.
  - RNR (1MXR) stays excluded: 1MXR is the R2 subunit. The H atom is abstracted by the
    Cys439 thiyl radical in R1, which carries no cofactor MEG-APSU could see.
- **test14 is development data from now on.** It was used to design the cobalamin, F430
  and PQQ rules. The cobalamin radius was first 8 Å and was set to 10 Å after diol
  dehydratase (1DIO, substrate at 8.1 Å from the cobalt) was missed.
- The scores of v1.2 and v1.3 are bit-identical to v1.5.1 on all 66 DEV and 21 test14
  structures; only the three labels changed.

## Development results

These are not tests, since both sets shaped the model.

| set | v1.6 | cofactor rule | EC rule | v1.6 vs cofactor rule | v1.6 vs EC rule |
|---|---|---|---|---|---|
| DEV, strong labels (46) | 100.0 % (92.3–100.0) | 80.4 % | 93.5 % | p = 0.004 | p = 0.25 |
| test14, strong labels (17) | 88.2 % (65.7–96.7) | 52.9 % | 94.1 % | p = 0.031 | p = 1.0 |
| test14, all labels (21) | 90.5 % (71.1–97.3) | 61.9 % | 81.0 % | p = 0.031 | p = 0.69 |

## Known before the run

- On 6 November 2026, while counting, the composition of test15 (6/29/19) and the label
  texts were seen. The names and PDB IDs are in the source code since 19 October 2026. In
  particular it was known that test15 contains a cobalamin transport protein. The
  cobalamin rule follows the chemistry above and was designed on DEV and test14. No model
  was run on a test15 structure before this tag.
- Glutamate dehydrogenase: an earlier note proposed replacing 1HWZ with 6DHD. The reason
  could not be reconstructed, so 1HWZ is kept.

## Procedure

1. This commit is tagged `test15-vorab` and pushed before the run.
2. One run from a fresh clone at the tag:
   `cargo build --release && ./target/release/meg-apsu bench test15`
3. The unedited output is committed as `test15_ergebnis.txt`.
