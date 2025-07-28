import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox

def parse_header(header_line):
    """Parse header line to extract grid parameters."""
    parts = header_line.strip().split()
    if len(parts) < 5:
        raise ValueError("Header must contain: G kx ky kz cutoff")
    kx_points = int(parts[1])
    ky_points = int(parts[2])
    length_cutoff = float(parts[4])
    return header_line, kx_points, ky_points, length_cutoff

def assign_weights(kpoints):
    """Assign weights to k-points based on their coordinates."""
    weights = []
    for kx, ky, kz in kpoints:
        # Special points get weight 1, others get weight 2
        if (abs(kx - 0.0) < 1e-6 and abs(ky - 0.0) < 1e-6) or \
           (abs(kx - 0.5) < 1e-6 and abs(ky - 0.0) < 1e-6) or \
           (abs(kx - 0.5) < 1e-6 and abs(ky - 0.5) < 1e-6):
            weights.append(1)
        else:
            weights.append(2)
    return weights

def remove_duplicates(kpoints, tolerance=1e-8):
    """Remove duplicate k-points with specified tolerance."""
    rounded = np.round(kpoints, decimals=8)
    _, idx = np.unique(rounded, axis=0, return_index=True)
    return kpoints[np.sort(idx)]

def generate_kpoints(header_line, kx_range, ky_range, kz_value, output_file, 
                    dirac_center=None, cone_density=None, cone_range=None):
    """Generate k-points mesh with optional Dirac cone enhancement."""
    
    # Parse header
    header, kx_points, ky_points, _ = header_line.strip().split()[:4]
    kx_points = int(kx_points)
    ky_points = int(ky_points)

    # Parse ranges
    kx_min, kx_max = map(float, kx_range.split(","))
    ky_min, ky_max = map(float, ky_range.split(","))

    # Generate base mesh
    kx_vals = np.linspace(kx_min, kx_max, kx_points)
    ky_vals = np.linspace(ky_min, ky_max, ky_points)
    kx_grid, ky_grid = np.meshgrid(kx_vals, ky_vals)
    kz_flat = np.full_like(kx_grid.flatten(), kz_value)

    kpoints = np.stack((kx_grid.flatten(), ky_grid.flatten(), kz_flat), axis=1)

    # Add Dirac cone enhancement if specified
    if dirac_center and cone_density and cone_range:
        center_coords = {
            "G": (0.0, 0.0),
            "K": (1/3, 1/3),
            "M": (0.5, 0.0),
        }
        
        dirac_center = dirac_center.upper()
        if dirac_center in center_coords:
            cx, cy = center_coords[dirac_center]
            r_min, r_max = map(float, cone_range.split(","))
            
            # Generate extra points around Dirac cone
            extra_kx = np.linspace(r_min, r_max, cone_density)
            extra_ky = np.linspace(r_min, r_max, cone_density)
            ex_kx_grid, ex_ky_grid = np.meshgrid(extra_kx, extra_ky)
            ex_kz_flat = np.full_like(ex_kx_grid.flatten(), kz_value)
            extra_kpoints = np.stack((ex_kx_grid.flatten(), ex_ky_grid.flatten(), ex_kz_flat), axis=1)
            
            # Combine base and extra points
            kpoints = np.vstack((kpoints, extra_kpoints))

    # Remove duplicates
    kpoints = remove_duplicates(kpoints)
    
    # Assign weights
    weights = assign_weights(kpoints)

    # Write output file
    with open(output_file, 'w') as f:
        f.write(f"{header_line}     # Parameters to Generate KPOINTS (Don't Edit This Line)\n")
        f.write(f"     {len(kpoints)}\n")
        f.write("Reciprocal lattice\n")
        for kp, w in zip(kpoints, weights):
            f.write(f"{kp[0]:16.12f} {kp[1]:16.12f} {kp[2]:16.12f} {w:4d}\n")

    # Optional visualization (requires matplotlib)
    try:
        plot_kpoints(kpoints, title=f"K-point Mesh ({len(kpoints)} points)")
    except ImportError:
        print("Matplotlib not available for visualization")

def plot_kpoints(kpoints, highlight=None, title="K-point Mesh"):
    """Plot k-points mesh (requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(kpoints[:, 0], kpoints[:, 1], s=10, color='black', label='K-points')
        
        if highlight is not None:
            ax.scatter(highlight[:, 0], highlight[:, 1], s=10, color='red', label='Cone region')
        
        ax.set_xlabel(r'$k_x$')
        ax.set_ylabel(r'$k_y$')
        ax.set_title(title)
        ax.grid(True)
        ax.legend()
        ax.set_aspect('equal')
        plt.show()
        
    except ImportError:
        raise ImportError("Matplotlib is required for visualization")

def launch_gui():
    """Launch the GUI for k-points generation."""
    root = tk.Tk()
    root.title("Gamma-Centered KPOINTS Generator")
    root.geometry("400x300")

    labels = [
        "Header (e.g. G 15 15 1 0.005):",
        "kx range (min,max):",
        "ky range (min,max):",
        "kz value:",
        "Dirac cone center (G, K, M) [optional]:",
        "Cone k-density [optional]:",
        "Cone k-range (min,max) [optional]:"
    ]
    entries = []

    for i, label in enumerate(labels):
        tk.Label(root, text=label).grid(row=i, column=0, sticky="e", padx=5, pady=2)
        e = tk.Entry(root, width=20)
        e.grid(row=i, column=1, padx=5, pady=2)
        entries.append(e)

    # Set default values
    entries[0].insert(0, "G 15 15 1 0.005")
    entries[1].insert(0, "0.0,1.0")
    entries[2].insert(0, "0.0,1.0")
    entries[3].insert(0, "0.0")

    def submit():
        try:
            header_line = entries[0].get().strip()
            kx_range = entries[1].get().strip()
            ky_range = entries[2].get().strip()
            kz_value = float(entries[3].get().strip())
            
            # Optional parameters
            dirac_center = entries[4].get().strip().upper() if entries[4].get().strip() else None
            cone_density = int(entries[5].get().strip()) if entries[5].get().strip() else None
            cone_range = entries[6].get().strip() if entries[6].get().strip() else None

            # Validate inputs
            if not header_line or not kx_range or not ky_range:
                raise ValueError("Please fill in all required fields")

            output_file = filedialog.asksaveasfilename(
                defaultextension=".kpoints", 
                filetypes=[("KPOINTS files", "*.kpoints"), ("All files", "*.*")]
            )
            if not output_file:
                return

            generate_kpoints(header_line, kx_range, ky_range, kz_value, output_file, 
                           dirac_center, cone_density, cone_range)
            
            messagebox.showinfo("Success", f"KPOINTS file saved to:\n{output_file}")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")

    tk.Button(root, text="Generate KPOINTS", command=submit, bg="lightblue").grid(
        row=len(labels), columnspan=2, pady=10)
    
    tk.Label(root, text="Note: Cone parameters are optional", font=("Arial", 8), fg="gray").grid(
        row=len(labels)+1, columnspan=2)

    root.mainloop()

if __name__ == "__main__":
    launch_gui()