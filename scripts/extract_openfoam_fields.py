"""
Extract all fields needed for RANS momentum from an OpenFOAM case directory.

Reads cell centers + velocity + pressure + turbulence fields from a single
OpenFOAM time directory (e.g. velocity_backup/808/) and writes a CSV with
columns: x, y, Ux, Uy, p, nut, k, epsilon.

Usage:
    python scripts/extract_openfoam_fields.py \
        --case case_834 \
        --time velocity_backup/808 \
        --case_id 834

Output goes to data/tiny/results/case_{case_id:03d}_full.csv.
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd


def parse_internal_field(filepath: str) -> np.ndarray:
    """
    Parse the `internalField` block from an OpenFOAM ASCII field file.

    Returns:
        (N,)   numpy array for volScalarField
        (N, 3) numpy array for volVectorField
    """
    with open(filepath) as f:
        text = f.read()

    is_vector = "List<vector>" in text
    is_scalar = "List<scalar>" in text
    if not (is_vector or is_scalar):
        raise ValueError(f"{filepath}: not a scalar or vector field "
                         "(or stored as 'uniform' — this parser only handles 'nonuniform')")

    m = re.search(r"nonuniform List<(scalar|vector)>\s*\n\s*(\d+)\s*\n\s*\(", text)
    if not m:
        raise ValueError(f"{filepath}: could not find 'nonuniform List<...>' block")
    n_expected = int(m.group(2))

    # Data lives between the '(' that opened the list and the matching ')'
    data_start = m.end()
    # Closing ')' followed by ';' may have intervening whitespace/newlines
    close_match = re.search(r"\)\s*;", text[data_start:])
    if not close_match:
        raise ValueError(f"{filepath}: could not find closing ');' of the list")
    data_text = text[data_start : data_start + close_match.start()]

    if is_vector:
        # Each line: '(a b c)'
        tuples = re.findall(r"\(\s*([^\)]+?)\s*\)", data_text)
        arr = np.array([[float(t) for t in tup.split()] for tup in tuples],
                       dtype=np.float64)
    else:
        arr = np.fromstring(data_text, sep="\n", dtype=np.float64)

    if arr.shape[0] != n_expected:
        raise ValueError(f"{filepath}: header says N={n_expected} but parsed "
                         f"{arr.shape[0]} values")
    return arr


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case",    required=True, help="Path to the OpenFOAM case directory")
    ap.add_argument("--time",    required=True, help="Time directory inside the case (e.g. 'velocity_backup/808' or '8940')")
    ap.add_argument("--case_id", type=int, required=True, help="Numeric case ID (used in output filename)")
    ap.add_argument("--outdir",  default=os.path.join("data", "tiny", "results"))
    ap.add_argument("--validate-against", default=None,
                    help="Optional: path to an existing CSV to cross-check x,y,Ux,Uy against")
    args = ap.parse_args()

    field_dir = os.path.join(args.case, args.time)
    if not os.path.isdir(field_dir):
        sys.exit(f"Directory not found: {field_dir}")

    # Required + optional fields
    required = ["Ccx", "Ccy", "U", "p", "nut"]
    optional = ["k", "epsilon"]

    data = {}
    for name in required:
        fp = os.path.join(field_dir, name)
        if not os.path.isfile(fp):
            sys.exit(f"Missing required field: {fp}")
        data[name] = parse_internal_field(fp)
        print(f"  parsed {name:8s}  shape={data[name].shape}")

    for name in optional:
        fp = os.path.join(field_dir, name)
        if os.path.isfile(fp):
            data[name] = parse_internal_field(fp)
            print(f"  parsed {name:8s}  shape={data[name].shape}")
        else:
            print(f"  skipped {name} (not found — optional)")
            data[name] = None

    # Sanity: all N's match
    n_cells = data["Ccx"].shape[0]
    for name, arr in data.items():
        if arr is None: continue
        if arr.shape[0] != n_cells:
            sys.exit(f"Cell-count mismatch: Ccx has {n_cells} but {name} has {arr.shape[0]}")

    # Build dataframe
    df_cols = {
        "x":  data["Ccx"],
        "y":  data["Ccy"],
        "Ux": data["U"][:, 0],
        "Uy": data["U"][:, 1],
        "p":  data["p"],
        "nut": data["nut"],
    }
    if data["k"]       is not None: df_cols["k"]       = data["k"]
    if data["epsilon"] is not None: df_cols["epsilon"] = data["epsilon"]
    df = pd.DataFrame(df_cols)

    # Optional cross-check
    if args.validate_against:
        ref = pd.read_csv(args.validate_against)
        if len(ref) != len(df):
            print(f"  [warn] row counts differ: existing CSV has {len(ref)}, parsed has {len(df)}")
        else:
            for col in ("x", "y", "Ux", "Uy"):
                if col in ref.columns:
                    max_abs_err = np.max(np.abs(ref[col].values - df[col].values))
                    status = "ok" if max_abs_err < 1e-5 else "MISMATCH"
                    print(f"  validate {col:3s}  max abs diff = {max_abs_err:.3e}  [{status}]")

    # Write out
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"case_{args.case_id:03d}_full.csv")
    df.to_csv(out, index=False)
    print(f"\nWrote {out}  ({len(df):,} rows, cols={list(df.columns)})")


if __name__ == "__main__":
    main()
