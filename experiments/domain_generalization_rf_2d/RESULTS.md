# Mixed-domain 2D result

The local RTX 2060 completed two matched 2,000-update fine-tunes. The target
mix was 50% ATLAS and 50% split evenly across EPISURG, EPIBIOS, and ARC. The
external clean-patch bank contains 1,792 training targets, while evaluation
uses 232 held-out targets: 32 each for EPISURG and EPIBIOS, and 168 ARC patches
from 14 sessions belonging to two held-out participants. Each patch is lesion-free within a 5-mm 3D
margin. Real labels supply only binary mask shape.

Metrics are means over two matched noise seeds. The three scores are masked
MAE (lower), masked gradient MAE (lower), and masked SSIM (higher).

| Held-out domain | ATLAS-only | Mixed, standard | Mixed, histogram + bias |
|---|---|---|---|
| EPISURG | 0.296 / 0.138 / 0.349 | **0.286 / 0.135 / 0.365** | 0.296 / 0.136 / 0.353 |
| EPIBIOS | 0.200 / 0.116 / 0.445 | **0.191 / 0.111 / 0.457** | 0.201 / 0.114 / 0.442 |
| ARC | 0.205 / 0.109 / 0.512 | **0.197 / 0.107 / 0.525** | 0.201 / 0.108 / 0.517 |
| ATLAS retention | 0.221 / 0.119 / 0.476 | **0.219 / 0.117 / 0.487** | 0.223 / 0.119 / 0.474 |

Mixed-domain clean targets improved all three added domains and slightly
improved the small ATLAS retention set. The improvement is modest but
consistent: masked MAE fell about 3–4% on EPISURG, EPIBIOS, and ARC. Adding the
tested histogram and bias-field augmentation removed that gain. That combined
augmentation should not be included in a longer run at these strengths.

This is stronger evidence for mixed-domain training than the preceding
ATLAS-only test, but the limits remain substantial: only two EPISURG/EPIBIOS
held-out cases, six ARC participants in the prepared local subset, and
synthetically hidden masks on already clean targets. The experiment does not
prove lesion recovery in real scans. A longer run is reasonable as the next
candidate, but should retain a held-out target-domain validation and compare
actual cases before treating the change as final.

See `domain_comparison.png` and `metrics.json` in the corresponding output
directory for the visual and per-domain records.
