"""
Environment smoke test.  Run from anywhere:
    python setup_check.py
"""

import sys, os


def main() -> int:
    print("== PIML Inpainting Demo Environment Check ==\n")
    try:
        import numpy, pandas, matplotlib, torch
        print(f"  [ok] numpy {numpy.__version__}, pandas {pandas.__version__}, "
              f"matplotlib {matplotlib.__version__}, torch {torch.__version__}")
    except ImportError as e:
        print(f"  [FAIL] {e}"); print("  Fix: pip install -r requirements.txt")
        return 1

    try:
        from piml.config import DataPaths, Hyperparams
        from piml.data   import load_case, make_sensor_split
        from piml.model  import MLP, get_device
        from piml.train  import train_mlp
    except ImportError as e:
        print(f"  [FAIL] piml: {e}"); return 1
    print("  [ok] piml package imports")

    dev = get_device()
    print(f"  [ok] device: {dev}")

    paths = DataPaths()
    if not os.path.exists(paths.geo_csv):
        print(f"  [FAIL] missing {paths.geo_csv}"); return 1
    geom, field = load_case(paths)

    split = make_sensor_split(field, n_sensors=30, seed=42)
    hp = Hyperparams(epochs=5, print_every=100_000, val_every=5)
    _ = train_mlp(split, geom, hp, dev, verbose=False)
    print("  [ok] 5-epoch MLP training ran to completion")

    print("\nAll checks passed. Open notebooks/01_sparse_data_setup.py to begin.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
