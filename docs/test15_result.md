# test15: result

Run once on 6 November 2026 at 18:31 CET, from a fresh clone at tag `test15-vorab`
(tree `6b473f4f`), as fixed in [`test15_protocol.md`](test15_protocol.md). The unedited
output is [`test15_ergebnis.txt`](../test15_ergebnis.txt) (SHA-256 `e7919ed2…78280`).
All 54 structures were evaluated.

## Primary analysis: strong labels

50 proteins: 4 tunnelling, 28 classical, 18 controls.

| model | accuracy (95 % CI) | tunnelling found | false alarms | TP / FN / TN / FP |
|---|---|---|---|---|
| MEG-APSU v1.6 | 70.0 % (56.2–80.9) | 4 of 4 | 15 of 46 | 4 / 0 / 31 / 15 |
| cofactor rule | 56.0 % (42.3–68.8) | 4 of 4 | 22 of 46 | 4 / 0 / 24 / 22 |
| EC rule | 100.0 % (92.9–100.0) | 4 of 4 | 0 of 46 | 4 / 0 / 46 / 0 |

Exact two-sided McNemar test on the discordant proteins:

- v1.6 against the cofactor rule: 7 proteins only v1.6 got right, 0 only the cofactor rule got right, p = 0.016.
- v1.6 against the EC rule: 0 and 15, p = 0.0001.

## Conclusions, as fixed in the protocol

- **Against the cofactor rule, v1.6 is significantly better.** On test15 the structure
  analysis adds information beyond cofactor presence.
- **Against the EC rule, v1.6 is significantly worse.** For annotated enzymes the EC number
  gives the better yes/no answer. The value of MEG-APSU therefore has to come from
  structures without annotation and from predicting the size of the isotope effect.

## Secondary results

- All labels (54 proteins): v1.6 72.2 % (59.1–82.4), cofactor rule 55.6 %, EC rule 96.3 %.
  The EC rule misses OleT and FNR, both with uncertain labels. McNemar p = 0.004 against
  the cofactor rule and p = 0.002 against the EC rule.
- v1.3 and v1.6 give the same score on 53 of 54 structures. The exception is M.HhaI (2HMY,
  classical, uncertain label): v1.3 scores it 48.4 through its SAM site, and v1.6 scores it
  0 after the SAM correction.
- KIE correlation: only 3 tunnelling proteins have a KIE value, too few to interpret.

## Where v1.6 errs

All 15 false alarms with strong labels are proteins without C–H chemistry:

- 11 of 18 controls: cupredoxins (amicyanin, stellacyanin, CopC), flavin photoreceptors
  (LOV2, VVD, BlrB), a Dps-like ferritin, mitoNEET, and haem or siderophore binding proteins
  (IsdE, CeuE, HmuT).
- 4 of 28 classical enzymes, all hydrolases: arginase, staphylococcal nuclease, peptide
  deformylase and methionine aminopeptidase.

These errors are not corrected on test15. Any correction has to be evaluated on a new
frozen set.
