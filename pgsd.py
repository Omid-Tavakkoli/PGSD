"""
Pore and Grain Size Distribution (PGSD) analysis for segmented micro-CT images.

Supports .raw (binary) and .tif/.tiff (ImageJ/tifffile stack) input formats.
Computes PGSD using the local thickness method (maximum inscribed sphere at each
voxel via Euclidean distance transform).
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage
from concurrent.futures import ThreadPoolExecutor
import os
import sys

try:
    import edt as _edt
    _HAS_EDT = True
except ImportError:
    _HAS_EDT = False
    print("Warning: 'edt' package not found. Install with: pip install edt\n"
          "         Falling back to scipy (single-threaded) for distance transform.")

try:
    import tifffile as _tifffile
    _HAS_TIFFFILE = True
except ImportError:
    _HAS_TIFFFILE = False


def read_input(filepath):
    """Parse the input.txt configuration file."""
    params = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '#' in line:
                line = line[:line.index('#')].strip()
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            if not value:
                continue
            if key in ('nx', 'ny', 'nz', 'pore_label', 'solid_label', 'num_threads'):
                params[key] = int(value)
            elif key == 'resolution':
                params[key] = float(value)
            elif key in ('cdf',):
                params[key] = value.lower() == 'true'
            else:
                params[key] = value
    return params


def load_raw(filename, nx, ny, nz):
    """Load a raw binary file as a 3D uint8 array shaped (nz, ny, nx)."""
    data = np.fromfile(filename, dtype=np.uint8)
    if data.size != nx * ny * nz:
        sys.exit(f"Error: file size {data.size} != {nx}*{ny}*{nz} = {nx*ny*nz}")
    return data.reshape((nz, ny, nx))


def load_tif(filename):
    """Load a TIFF stack as a 3D array. Requires tifffile."""
    if not _HAS_TIFFFILE:
        sys.exit("Error: reading .tif files requires tifffile. Install with: pip install tifffile")
    image = _tifffile.imread(filename)
    if image.ndim == 2:
        image = image[np.newaxis, ...]   # single-slice → (1, ny, nx)
    if image.ndim != 3:
        sys.exit(f"Error: expected a 3-D TIFF stack, got shape {image.shape}")
    return image


def load_image(filename, params):
    """Dispatch to the correct loader based on file extension."""
    ext = os.path.splitext(filename)[1].lower()
    if ext in ('.tif', '.tiff'):
        image = load_tif(filename)
        nz, ny, nx = image.shape
        print(f"Loading {os.path.basename(filename)} ({nx}x{ny}x{nz}, from TIFF)...")
        return image
    elif ext == '.raw':
        for dim in ('nx', 'ny', 'nz'):
            if dim not in params:
                sys.exit(f"Error: '{dim}' is required in input.txt for .raw files.")
        nx, ny, nz = params['nx'], params['ny'], params['nz']
        print(f"Loading {os.path.basename(filename)} ({nx}x{ny}x{nz}, from RAW)...")
        return load_raw(filename, nx, ny, nz)
    else:
        sys.exit(f"Error: unsupported file format '{ext}'. Use .raw, .tif, or .tiff.")


def _parallel_max_filter(dist, num_threads):
    """maximum_filter(size=3) split across z-slices using threads."""
    if num_threads <= 1:
        return ndimage.maximum_filter(dist, size=3)

    nz = dist.shape[0]
    chunk_size = max(3, (nz + num_threads - 1) // num_threads)
    slices = [(z, min(z + chunk_size, nz)) for z in range(0, nz, chunk_size)]

    def _process(z_z_end):
        z, z_end = z_z_end
        z_lo, z_hi = max(0, z - 1), min(nz, z_end + 1)
        filtered = ndimage.maximum_filter(dist[z_lo:z_hi], size=3)
        inner = z - z_lo
        return filtered[inner: inner + (z_end - z)]

    result = np.empty_like(dist)
    with ThreadPoolExecutor(max_workers=num_threads) as ex:
        for (z, z_end), chunk in zip(slices, ex.map(_process, slices)):
            result[z:z_end] = chunk
    return result


def compute_pgsd(image, solid_label, measure='pore', num_threads=1):
    """
    Compute size distribution using volume-weighted local maxima of the EDT.

    measure='pore'  — operates on the pore phase (image != solid_label)
    measure='grain' — operates on the solid/grain phase (image == solid_label)
    """
    if measure == 'grain':
        phase_mask = image == solid_label
        phase_frac = np.sum(phase_mask) / phase_mask.size * 100
        print(f"Solid fraction: {phase_frac:.2f}%")
        print(f"Solid voxels: {np.sum(phase_mask):,}")
    else:
        phase_mask = image != solid_label
        phase_frac = np.sum(phase_mask) / phase_mask.size * 100
        print(f"Porosity: {phase_frac:.2f}%")
        print(f"Pore voxels: {np.sum(phase_mask):,}")

    print(f"Computing Euclidean distance transform (threads={num_threads})...")
    if _HAS_EDT:
        dist = _edt.edt(phase_mask, parallel=num_threads)
    else:
        dist = ndimage.distance_transform_edt(phase_mask)

    print("Finding local maxima...")
    dilated = _parallel_max_filter(dist, num_threads)
    local_max_mask = (dist == dilated) & (dist > 0)
    radii_at_maxima = dist[local_max_mask]
    print(f"Local maxima found: {radii_at_maxima.size:,}")

    weights = radii_at_maxima ** 3

    max_radius = int(np.max(radii_at_maxima))
    bins = np.arange(0.5, max_radius + 1.5, 1)
    radii = np.arange(1, max_radius + 1)

    psd, _ = np.histogram(radii_at_maxima, bins=bins, weights=weights)

    total = np.sum(psd)
    if total > 0:
        psd = psd / total

    return radii, psd, phase_frac


def plot_pgsd(radii, psd, porosity, output_path, resolution=1.0, cdf=True, measure='pore'):
    """Plot size distribution as bar chart and optionally cumulative curve."""
    sns.set(style="white")
    palette = sns.color_palette(
        ["#000000", "#CC79A7", "#56B4E9", "#E69F00", "#009E73", "#F0E442", "#0072B2", "#D55E00"]
    )

    n_target = 50
    factor = max(1, int(np.ceil(len(radii) / n_target)))
    if factor > 1:
        trim = (len(radii) // factor) * factor
        radii = radii[:trim].reshape(-1, factor).mean(axis=1)
        psd   = psd[:trim].reshape(-1, factor).sum(axis=1)

    radii_um = radii * resolution
    bar_width = 0.8 * factor * resolution
    bar_alpha = 0.22 if cdf else 1.0

    if measure == 'grain':
        x_label   = 'Grain radius [μm]'
        bar_label = 'GSD'
        title     = f'Grain Size Distribution (solid fraction = {porosity:.1f}%)'
    else:
        x_label   = 'Pore radius [μm]'
        bar_label = 'PSD'
        title     = f'Pore Size Distribution (porosity = {porosity:.1f}%)'

    fig, ax1 = plt.subplots(figsize=(7, 5))

    ax1.bar(
        radii_um, psd, width=bar_width, color=palette[2], alpha=bar_alpha,
        edgecolor=palette[2], linewidth=0.5, label=bar_label
    )
    ax1.set_xlabel(x_label)
    ax1.set_ylabel('Volume fraction')

    if cdf:
        ax2 = ax1.twinx()
        cdf_vals = np.cumsum(psd)
        ax2.plot(
            radii_um, cdf_vals,
            color=palette[7], linewidth=1.5, alpha=0.9, label='CDF',
        )
        ax2.set_ylabel('Cumulative fraction', color=palette[7])
        ax2.tick_params(axis='y', labelcolor=palette[7])
        ax2.set_ylim(0, 1.05)
        for spine in ax2.spines.values():
            spine.set_visible(True)

    ax1.set_title(title)

    for spine in ax1.spines.values():
        spine.set_visible(True)

    lines1, labels1 = ax1.get_legend_handles_labels()
    if cdf:
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines1 += lines2
        labels1 += labels2
    ax1.legend(lines1, labels1, loc="best")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")
    plt.show()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, 'input.txt')

    params = read_input(input_file)
    filepath = os.path.join(script_dir, params['filename'])

    image = load_image(filepath, params)

    measure = params.get('measure', 'pore').lower()
    if measure not in ('pore', 'grain'):
        sys.exit(f"Error: measure must be 'pore' or 'grain', got '{measure}'")

    num_threads = params.get('num_threads', 1)
    radii, psd, phase_frac = compute_pgsd(image, params['solid_label'], measure=measure,
                                           num_threads=num_threads)

    resolution = params.get('resolution', 1.0)
    cdf = params.get('cdf', True)

    output_path = os.path.join(script_dir, params.get('save_fig', 'pgsd_plot.pdf'))
    plot_pgsd(radii, psd, phase_frac, output_path, resolution=resolution, cdf=cdf, measure=measure)

    res = resolution
    mean_r     = np.sum(radii * psd) * res
    median_idx = np.searchsorted(np.cumsum(psd), 0.5)
    median_r   = radii[min(median_idx, len(radii) - 1)] * res
    mode_r     = radii[np.argmax(psd)] * res
    max_r      = radii[np.max(np.nonzero(psd))] * res
    label      = 'grain' if measure == 'grain' else 'pore'
    print(f"\n{'GSD' if measure == 'grain' else 'PSD'} Summary (resolution = {res} µm/voxel):")
    print(f"  Mode {label} radius:   {mode_r:.1f} µm")
    print(f"  Mean {label} radius:   {mean_r:.1f} µm")
    print(f"  Median {label} radius: {median_r:.1f} µm")
    print(f"  Max {label} radius:    {max_r:.1f} µm")


if __name__ == '__main__':
    main()
