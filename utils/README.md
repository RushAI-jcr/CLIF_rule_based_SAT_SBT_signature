# Utils Directory

Shared utility modules for the SAT/SBT CLIF project. All analysis scripts in `code/` import from this directory via `sys.path`.

## Modules

| Module | Purpose |
|---|---|
| `config.py` | Loads `config/config.json` for site-specific settings |
| `config.R` | R equivalent of config loader |
| `pyCLIF.py` | CLIF data loading, datetime handling, respiratory waterfall processing |
| `pyCLIF2.py` | Extended CLIF utilities |
| `pySBT.py` | SBT phenotyping helpers, TableOne formatting functions |
| `pySofa.py` | SOFA score computation from CLIF tables |
| `definitions_source_of_truth.py` | **Single source of truth** for all phenotype definitions, thresholds, and study parameters |
| `site_output_schema.py` | Schema definition and validation for federated site-level output CSVs |
| `meta_analysis.py` | Two-stage meta-analysis: logit-transformed proportion pooling, forest/funnel plots |
| `outlier_handler.R` | R functions for applying outlier thresholds from `outlier-thresholds/` CSVs |

## Import Pattern

From `code/` directory (notebooks or scripts):
```python
import sys, os
sys.path.insert(0, os.path.join(os.pardir, "utils"))

import pyCLIF as pc
from definitions_source_of_truth import SAT_SEDATIVES, SBT_CONTROLLED_MODES
```
