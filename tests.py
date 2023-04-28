from gedAPI import GedLoader
from ingester3.extensions import ViewsMonth

assert GedLoader('21.4').exists is False
assert GedLoader('22.1').exists is True
assert GedLoader('22.0.1').exists is True
assert GedLoader('22.0.1').min_month == GedLoader('22.0.1').max_month
assert GedLoader('22.1').min_month != GedLoader('22.1').max_month
assert GedLoader('22.1').min_month == ViewsMonth.from_year_month(1989, 1)
