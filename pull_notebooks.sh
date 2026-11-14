#!/usr/bin/env bash
# Downloads all your Kaggle notebooks for this project and sorts them into the repo's folders.
# Needs: pip install kaggle, and ~/.kaggle/kaggle.json (Kaggle -> Settings -> API -> Create New Token).
# Run from the repo root:  bash pull_notebooks.sh
set -euo pipefail
RAW=notebooks/_kaggle_raw
mkdir -p "$RAW"

# 1. Pull every notebook you own (code only, no outputs or data)
kaggle kernels list --mine --page-size 100 --csv | tail -n +2 | cut -d, -f1 | while read -r ref; do
  echo "pulling $ref"
  kaggle kernels pull "$ref" -p "$RAW/$(basename "$ref")" || echo "  skipped $ref"
done

# 2. Sort them by the number at the start of each notebook's name
python3 - <<'PY'
import os, shutil, glob
raw = 'notebooks/_kaggle_raw'
where = {
  '02': '01_data',
  '14': '02_training', '14aug': '02_training', '15': '02_training', '17': '02_training', '17v2': '02_training',
  '18': '03_aggregator', '18aug': '03_aggregator', '19b': '03_aggregator', '19e': '03_aggregator',
  '16e': '04_submission',
}
experiments = {'05','06','07','08','09','10','11','12','13','16','16b','16c','16d','17e','18v2','19','19c','20'}
for d in sorted(os.listdir(raw)):
    files = glob.glob(f'{raw}/{d}/*.ipynb')
    if not files:
        continue
    key = d.split('-')[0].lower()
    folder = where.get(key) or ('experiments' if key in experiments else '_unsorted')
    os.makedirs(f'notebooks/{folder}', exist_ok=True)
    dst = f'notebooks/{folder}/{d.replace("-", "_")}.ipynb'
    shutil.copy(files[0], dst)
    print(f'{d:35s} -> {dst}')
print('\nAnything in notebooks/_unsorted needs a folder by hand. Delete notebooks/_kaggle_raw when done.')
PY
