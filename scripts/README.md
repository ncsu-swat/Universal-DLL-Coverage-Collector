## How to run

Safe test (no changes):
Optional
python3 classify_torch_valid_by_api.py --dry-run --max-files 20
Full classification (symlinks into a new tree):
Optional
python3 classify_torch_valid_by_api.py
Alternatives:
Optional
python3 classify_torch_valid_by_api.py --mode copy
Optional
python3 classify_torch_valid_by_api.py --mode hardlink
Optional
python3 classify_torch_valid_by_api.py --mode move
Index with file lists (can be large):
Optional
python3 classify_torch_valid_by_api.py --full-index

## What it creates

Results/torch/valid_by_api/<api path>/
Example: torch.nn.functional.leaky_relu_1291.py (symlink by default)
Results/torch/valid_by_api/index.csv
Rows: api,count
Results/torch/valid_by_api/index.json
Summary counts; includes per-file lists if --full-index is used
