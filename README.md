# ETCalcPyMOL: Electron Transfer Calculator for PyMOL

[![PyMOL](https://img.shields.io/badge/PyMOL-Plugin-blue)](https://pymol.org/) [![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)](https://www.python.org/) [![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

ETCalcPyMOL is a PyMOL plugin (v0.1) for computing electron transfer (ET) rates in biomolecules. It supports classical Marcus theory and advanced extensions like Marcus-Levich-Jortner (MLJ), Redfield, Trajectory Surface Hopping (TSH), and empirical models. The tool analyzes static structures or dynamic MD trajectories, visualizes ET paths in PyMOL, and includes a GUI for easy parameter tuning.

Built on NumPy, SciPy, NetworkX, MDAnalysis, and Tkinter, it enables ET network detection, parameter calibration, and bridge-mediated calculations.

## Features

- **ET Rate Theories**:
  - Classical Marcus: Distance-dependent coupling with orientation factors.
  - MLJ: Vibronic effects with single/multi-mode support.
  - Redfield: Quantum dephasing in weak-coupling limit.
  - TSH: Semiclassical surface hopping with frustration statistics.
  - Empirical Marcus-Moser-Dutton hybrid.

- **Geometry & Coupling**:
  - Edge-edge distances for redox moieties (e.g., TRP, TYR, PHE, HIS, FAD, AMP).
  - Orientation factors (κ²) from ring planes.
  - Aromatic metrics: Centroid distances, min edge distances, plane angles.

- **Bridge-Mediated ET**:
  - Static: Constant, potential-based, Green's function (GF), hopping.
  - Dynamic (MD): Time-averaged rates with segment correlations.

- **Network Analysis**:
  - Detect ET paths in proteins.
  - Calibrate H0, β, λ to target rates with bounds and plots.

- **MD Integration**:
  - Analyze Amber trajectories (.prmtop + .nc).
  - Kubo-Anderson correction for disorder.

- **GUI**: Tkinter-based interface for all commands, with collapsible theory-specific params.

- **Visualization**: PyMOL arrows, distances, centroids, normals for ET pairs/paths.

## Installation

1. **Dependencies**:
   - PyMOL (v2.0+)
   - Python libraries: `numpy`, `scipy`, `matplotlib`, `networkx`, `MDAnalysis`, `tkinter` (built-in), `pymol.cmd` & `pymol.cgo`.
   - Install via pip:  
     ```
     pip install numpy scipy matplotlib networkx MDAnalysis
     ```

2. **Plugin Setup**:
   - Download `ETCalcPyMOL.py` and place it in your PyMOL scripts directory (e.g., `~/.pymol/startup/`).
   - Load in PyMOL:  
     ```
     run ETCalcPyMOL.py
     ```
   - Or install as a plugin via PyMOL's Plugin Manager: File > Plugin > Install Plugin.

3. **GUI Launch**:  
   ```
   et_plugin_gui
   ```

## Usage

ETCalcPyMOL extends PyMOL with new commands. Defaults: λ=0.70 eV, ΔG=-0.50 eV, T=298.15 K, H0=0.050 eV, β=1.10 Å⁻¹, R0=3.50 Å.

### Basic ET Rate (Marcus)
```
et_rate_marcus default, 1DNP, A, 382, A, 472, 0.70, -0.50, 298.15, 0.050, 1.10, 3.50, on, 0.00, 1.00
```
- Computes rate, visualizes arrow/distance.
- Orient: `on/off`, floor (min κ), pow (exponent).

### MLJ Rate
```
et_rate_MLJ default, 1DNP, A, 382, A, 472, 0.70, -0.50, 298.15, 0.050, 1.10, 3.50, 1.0, 0.18, 10, on, 0.00, 1.00
```
- Supports multi-mode via CSV: S_csv="1.0,0.5", hw_csv="0.18,0.20".

### Aromatic Metrics
```
et_aromatic_metrics 1DNP, A, 382, A, 472, default
```
- Computes distances, angles; visualizes sticks, spheres, arrows.

### Static Bridge ET
```
et_bridge_chain_const default, 1DNP, A, 306, A, 472, A:359+A:382, 0.70, -0.50, 298.15, 0.050, 1.10, 3.50, 1.00, on, 0.00, 1.00
```
- Modes: `const` (ΔB), `pot` (EoxD, EredA, Ereds), `gf` (Etun, ΔBs), `hop` (dGs).

### Dynamic ET (MD)
```
et_rate_marcus_MD topology.prmtop, trajectory.nc, A, 382, A, 472, 0.70, -0.50, 298.15, 0.050, 1.10, 3.50, on, 0.00, 1.00, default, True, True
```
- Plots histogram; applies Kubo correction.

### Dynamic Bridge ET (MD)
```
et_bridge_chain_dynamic_MD topology.prmtop, trajectory.nc, A, 306, A:359+A:382, A, 472, const, pot, 0.0,-0.7,-0.5,-0.4, 0.70, -0.50, 298.15, 0.050, 1.10, 3.50, on, 0.00, 1.00, default, True, True, True
```
- Segment stats & correlation heatmap.

### Network Detection
```
et_detect_network_path 1DNP, 12.0, marcus, lambda_eV=0.70 deltaG_eV=-0.50 T_K=298.15 H0=0.050 beta=1.10 R0=3.50, donor=A:382, acceptor=A:472, on, 0.00, 1.00, True
```

### Calibration
```
calibrate_Marcus_to_k 1DNP, A, 382, A, 472, marcus, lambda_eV=0.70 deltaG_eV=-0.50 T_K=298.15 H0=0.050 beta=1.10 R0=3.50, 1.25e12, H0+lambda, Yes, fit_range_H0=0.001,0.5 fit_range_lambda=0.1,2.0 fit_range_beta=0.1,2.0, default, on, 0.00, 1.00, on
```

### GUI
Launch with `et_plugin_gui`. Tabs for detection, calibration, rates, bridges, MD, metrics.

## Examples

- **Static Marcus**: Visualize ET from TRP382 to FAD472 in 1DNP.
- **MD Analysis**: Average rates over 100 ns trajectory.
- **Bridge Hop**: Multi-step hopping with fitted ΔG per bridge.
- **Fit Params**: Optimize H0/β to match experimental k=10¹² s⁻¹.

See code comments/USAGE strings for details.

## Contributing

Pull requests welcome! For issues, open a GitHub issue.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Credits

Developed using PyMOL API, inspired by Marcus/Redfield/TSH theories. Contact: [Your Email/GitHub].
