## PGSD

This repository computes Pore and Grain Size Distribution (PGSD) from a segmented 3D micro-CT image using the local thickness method (maximum inscribed sphere at each voxel via Euclidean distance transform).

The script supports:
- `.raw` volumes (with dimensions from `input.txt`)
- `.tif` / `.tiff` 3D stacks (dimensions read from file)

---

### Installation

1) Python 3.9+ recommended.
2) Install dependencies:

- **pip**:

```bash
pip install numpy matplotlib seaborn scipy tifffile edt
```

- **conda**:

```bash
conda create -n pgsd python=3.10 -y
conda activate pgsd
conda install -c conda-forge numpy matplotlib seaborn scipy tifffile -y
pip install edt
```

Note: `edt` is optional but strongly recommended — it replaces the single-threaded scipy distance transform with a fast multi-threaded implementation. Without it, the code falls back to scipy automatically.

---

### Inputs

- `pgsd.py`: main script
- `input.txt`: user-editable configuration file

---

### How to Run

```bash
python3 pgsd.py
```

The script reads `input.txt` from the same directory as `pgsd.py`.

---

### Configuration (`input.txt`)

**1) Input file**

- `filename`: input image path (`.raw`, `.tif`, `.tiff`)

**2) RAW file dimensions (required only for `.raw`)**

- `nx`, `ny`, `nz`: volume dimensions in voxels
- For `.tif` / `.tiff`, these are read directly from the file and must be omitted.

**3) Phase labels**

- `pore_label`: voxel value representing pore space
- `solid_label`: voxel value representing solid/grain phase
- For non-binary images, all voxel values other than `solid_label` are treated as pore.

**4) Resolution**

- `resolution`: voxel size in micrometers (µm/voxel), used to convert radii from voxels to µm in the plot and summary statistics.

**5) Measure**

- `measure`: which phase to analyse
  - `pore` — computes Pore Size Distribution (PSD)
  - `grain` — computes Grain Size Distribution (GSD)

**6) Performance**

- `num_threads`: number of CPU threads for the distance transform and maximum filter (requires `edt`; ignored otherwise)

**7) Output settings**

- `cdf`: set to `true` to overlay a cumulative distribution curve on the plot
- `save_fig`: output figure path (`.pdf` or `.png`)

---

### Outputs

**1. Figure (`save_fig`)**:
   - Volume-weighted size distribution as a bar chart
   - Optional cumulative distribution function (CDF) curve on a secondary axis

**2. Console summary**:
   - Porosity or solid fraction (%)
   - Mode, mean, median, and maximum pore/grain radius (µm)

---

### Method

The local thickness method assigns to each pore (or grain) voxel the radius of the largest sphere that fits inside the phase and contains that voxel. This is computed efficiently as follows:

1. Compute the Euclidean distance transform (EDT) of the phase mask — each voxel receives its distance to the nearest opposite-phase voxel.
2. Find local maxima of the EDT — these are the sphere centres (inscribed sphere radii).
3. Build a volume-weighted histogram: each centre contributes weight proportional to *r³* (sphere volume), so the distribution reflects the fraction of phase volume occupied by pores/grains of each size.
