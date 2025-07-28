#!/usr/bin/env python3
"""
Fixed version: Generates unsorted KX.grd, KY.grd to match EIGENVAL and BAND grid order.
"""

import numpy as np

def parse_kpoints_file(filename="KPOINTS"):
    with open(filename, 'r') as f:
        lines = f.readlines()

    kpoint_count = None
    start_line = None
    for i, line in enumerate(lines):
        if line.strip().isdigit():
            kpoint_count = int(line.strip())
            start_line = i + 2
            break

    if kpoint_count is None:
        raise ValueError("Could not find number of k-points in KPOINTS file")

    kpoints = []
    for i in range(start_line, start_line + kpoint_count):
        coords = lines[i].strip().split()[:3]
        kx, ky, kz = map(float, coords)
        kpoints.append((kx, ky, kz))

    return np.array(kpoints)

def save_grid_file(grid, filename):
    with open(filename, 'w') as f:
        for row in grid:
            f.write("".join(f"{val:10.4f}" for val in row) + "\n")

def read_fermi_energy(fermi_file="FERMI_ENERGY"):
    try:
        with open(fermi_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    return float(line.split()[0])
    except:
        pass
    return None

def parse_eigenval_for_bands(eigenval_file="EIGENVAL", fermi_energy=None):
    try:
        with open(eigenval_file, 'r') as f:
            lines = f.readlines()

        header = lines[5].split()
        nelect = int(float(header[0]))
        nkpts = int(header[1])
        nbands = int(header[2])

        homo_index = nelect // 2
        lumo_index = homo_index + 1

        if fermi_energy is None:
            fermi_energy = 0.0

        homo_energies = []
        lumo_energies = []
        idx = 7

        for _ in range(nkpts):
            while not lines[idx].strip():
                idx += 1
            idx += 1
            for b in range(nbands):
                parts = lines[idx].split()
                band = int(parts[0])
                energy = float(parts[1]) - fermi_energy
                if band == homo_index:
                    homo_energies.append(energy)
                elif band == lumo_index:
                    lumo_energies.append(energy)
                idx += 1

        return np.array(homo_energies), np.array(lumo_energies)

    except:
        return None, None

def main():
    print("Reading KPOINTS...")
    kpoints = parse_kpoints_file()
    print(f"Loaded {len(kpoints)} k-points")

    # Preserve order as in KPOINTS
    kx = kpoints[:, 0].reshape(-1, 1)
    ky = kpoints[:, 1].reshape(-1, 1)

    save_grid_file(kx, "KX.grd")
    save_grid_file(ky, "KY.grd")
    print("Saved KX.grd and KY.grd (unsorted)")

    fermi = read_fermi_energy()
    homo, lumo = parse_eigenval_for_bands(fermi_energy=fermi)

    if homo is not None and lumo is not None:
        save_grid_file(homo.reshape(-1, 1), "BAND_HOMO.grd")
        save_grid_file(lumo.reshape(-1, 1), "BAND_LUMO.grd")
        print("Saved BAND_HOMO.grd and BAND_LUMO.grd")
    else:
        print("Could not create band energy grids (EIGENVAL missing or invalid)")

if __name__ == "__main__":
    main()
