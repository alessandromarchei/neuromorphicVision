#!/usr/bin/env python3
import h5py
import sys

def print_hdf5_structure(hdf5_file):
    """
    Stampa tutti i gruppi, dataset e shape presenti in un file HDF5.
    """
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"[DATASET] {name}")
            print(f"  - shape: {obj.shape}")
            print(f"  - dtype: {obj.dtype}")
            print("")
        elif isinstance(obj, h5py.Group):
            print(f"[GROUP]   {name}")

    with h5py.File(hdf5_file, "r") as f:
        print(f"\nInspecting HDF5 file: {hdf5_file}\n")
        f.visititems(visit)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_hdf5.py <file.hdf5>")
        sys.exit(1)

    hdf5_path = sys.argv[1]
    print_hdf5_structure(hdf5_path)
