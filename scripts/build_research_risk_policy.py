"""Generar research provisional desde originales confiables; solo archivos nuevos."""

import argparse
import json
from pathlib import Path

from depi_ml.analysis.research_risk import build


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--manifest-sha256",required=True)
    parser.add_argument("--csv",type=Path,default=Path("data/exports/training_dataset_v1.csv"))
    parser.add_argument("--labels",type=Path,default=Path("data/exports/training_dataset_v1_labels.csv"))
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    bundle=build(args.reference,args.manifest_sha256,args.csv,args.labels,args.output)
    print(json.dumps([{k:p[k] for k in ["model_key","low_upper_threshold","high_lower_threshold","policy_hash_sha256","support_passed"]}
                      for p in bundle["policies"]],indent=2))


if __name__=="__main__":
    main()
