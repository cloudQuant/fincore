# `fincore.portfolio`

Portfolio operations are split by data responsibility:

| Module | Responsibility |
| --- | --- |
| `positions` | exposures and leverage |
| `transactions` | transaction normalisation and turnover |
| `round_trips` | matched trades and round-trip analysis |
| `capacity` | liquidity and capacity calculations |
| `models` | immutable portfolio domain models |
| `operations` | registered direct operations |

The report domain consumes these canonical inputs; it does not invoke a
portfolio façade.

```python
from fincore.portfolio.positions import gross_lev
from fincore.portfolio.transactions import get_turnover
```
