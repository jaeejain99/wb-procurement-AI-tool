"""Analyse a package on REAL data after `python -m procintel.live_wb --out data/live`.
   python run_live.py data/live "Rehabilitation of rural roads" "Civil Works" Transport KE 20 ICB"""
import sys, warnings; warnings.filterwarnings("ignore")
from procintel import Copilot, PackageSpec, Platform, render
d, desc, cat, sector, cc, val, meth = (sys.argv[1:] + [None] * 7)[:7]
plat = Platform.build(d or "data/live")
print("data quality:", plat.ds.quality, "\nmodel holdout:", {k: round(v, 3) for k, v in plat.models.metrics.items()})
print(render(Copilot(plat).analyze(PackageSpec(desc or "Rehabilitation of rural roads", cat or "Civil Works", sector or "Transport", cc or "KE", float(val or 20), meth or "International Competitive Bidding"))))
