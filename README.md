# Metal complex builder (molSimplify + RDKit)

A Jupyter notebook that builds 3D starting structures of transition-metal complexes reacting with
aryl electrophiles (Ar–Cl, Ar–Br, Ar–I, Ar–OTf, Ar–OTs, ...) and writes ready-to-run Gaussian 16 +
xTB inputs (via [`xtb-gaussian`](https://github.com/aspuru-guzik-group/xtb-gaussian)) plus a SLURM
array script.

You give it the metal, its oxidation state, the spectator ligands, and a list of aryl electrophiles
as SMILES. It builds the starting complex, the products of several mechanistic pathways, every
distinct isomer, and inputs for every possible spin state, and numbers the reacting atoms
consistently so the next steps (scans, TS searches) are easy to set up.

**The structures are starting geometries only.** molSimplify places ligands on geometry templates
using tabulated bond lengths. Every structure must be optimized (e.g. with g-xTB), checked for zero
imaginary frequencies, and inspected before use.

## What it builds

For a starting complex **M(n)(L)** and an electrophile **Ar–X**:

| Channel | Metal species | Partner species |
|---|---|---|
| `start` (always) | M(n)(L) | Ar–X |
| `OA` | M(n+2)(L)(Ar)(X): two-electron oxidative addition (concerted) | — |
| `OA_cation` | [M(n+2)(L)(Ar)]⁺: ionic / SNAr-like oxidative addition | X⁻ |
| `XAT` | M(n+1)(L)(X): halogen/group-atom abstraction, inner-sphere ET | Ar• |
| `OSET` | [M(n+1)(L)]⁺: outer-sphere electron transfer (start geometry, one electron removed) | [Ar–X]•⁻ |

Along the way the notebook:

- **Splits each Ar–X** into an aryl ligand (ipso carbon written as atom 1, so molSimplify binds it
  there) and a leaving group. Halides use molSimplify's library entries; sulfonates (OTf, OTs, OMs,
  ONf, and other Ar–O–SO₂R) are passed as SMILES bound through O.
- **Reads ligand charges and denticities** from molSimplify's ligand library, or from RDKit for
  SMILES ligands, and computes each species' total charge.
- **Chooses the geometry** from the coordination number (2 `li`, 3 `tpl`, 4 `sqp`, 5 `spy`,
  6 `oct`, 7 `pbp`), unless you override it.
- **Enumerates isomers** by building different orderings of the monodentate ligands, then keeps one
  structure per distinct trans-ligand pattern (repeats are marked `duplicate`).
- **Writes all spin states** allowed by the metal's d-electron count (e.g. doublet and quartet for
  Ni(III)), and skips any multiplicity that is impossible for the electron count.
- **Checks every structure**: coordination number, metal–ligand distances, atom clashes, and which
  ligands are trans to each other.
- **Renumbers atoms** so the metal is atom 1, the ipso carbon atom 2, and the leaving-group atom
  bound to the metal atom 3. A C–X scan from an `OA` product is then always `B 2 3 S <n> -0.05`.
- **Flags isomers with X trans to Ar**, which a three-centered concerted oxidative addition cannot
  form directly.

## Requirements

- Python 3.10, Jupyter (or VS Code with the Jupyter extension)
- [molSimplify](https://github.com/hjkgrp/molSimplify) (tested with 2.0.0), RDKit, OpenBabel, pandas
- Optional: py3Dmol, for viewing structures inside the notebook
- To run the generated inputs: Gaussian 16, xTB, `xtb-gaussian`, and SLURM

## Installation

molSimplify is **not** available from conda-forge under that name; install it with pip inside a
conda environment:

```bash
mamba create -n msimp -c conda-forge python=3.10 rdkit openbabel ipykernel pandas
conda activate msimp
pip install molSimplify py3Dmol
python -m ipykernel install --user --name msimp
```

(`conda create` works the same as `mamba create`.) `pip install molSimplify` also installs
TensorFlow, which is large. The notebook doesn't need it (`SKIP_ANN = True`); if the TensorFlow
install fails, use `pip install --no-deps molSimplify` and then `pip install importlib_resources
networkx scipy`, adding any other module reported missing.

Open `Metal_complex_builder.ipynb`, select the `msimp` kernel, and run the cells in order.

**Linux or WSL is recommended.** On native Windows, molSimplify 2.0.0 fails with
`PermissionError: [WinError 32] ... being used by another process` because it deletes a temporary
file that OpenBabel still has open. Either run the notebook in WSL, or patch the installed
`molSimplify/Classes/mol3D.py` so that the `os.remove(tempf)` in `convert2OBMol` is wrapped in
`try: ... except PermissionError: pass`. All files the notebook writes use Unix line endings, even
on Windows.

## Using the notebook

### 1. Settings

| Variable | Default | Meaning |
|---|---|---|
| `WORKDIR` | `complex_structures` | Output folder |
| `MS_CMD` | `python -m molSimplify` | How molSimplify is called |
| `SKIP_ANN` | `True` | Use tabulated bond lengths instead of molSimplify's neural network |
| `ISOMERS` | `"all"` | `"all"` tries every distinct monodentate ordering; `"first"` only varies the first (faster) |
| `MAX_ORDERINGS` | `12` | Maximum molSimplify builds per species |
| `NPROC`, `MEM` | `4`, `16000MB` | Threads (`-P`) and Gaussian memory; match the SLURM script |
| `XTB_METHOD` | `--gxtb` | xTB method (e.g. `--gfn2`); never combine methods |
| `XTB_ACC` | `0.0001` | xTB accuracy |
| `XTB_EXTRA` | `""` | Extra xTB flags, e.g. `--alpb dmf` if supported by your xTB build for this method |

Keep the xTB settings identical for every structure you intend to compare.

### 2. Inputs (section 4)

```python
LIGANDS = {                                   # spectator ligands only
    "terpy": {"library": "terpy"},
    # "myL": {"smiles": "c1ccc(nc1)-c1cccc(n1)-c1ccccn1", "donors": [5, 12, 18]},
    "Br":    {"library": "bromide"},
    "PPh3":  {"library": "pph3"},
}

SYSTEMS = [
    dict(name="terpyNiBr", metal="Ni", ox=1, spectators=[("terpy", 1), ("Br", 1)]),
    dict(name="PdPPh3",    metal="Pd", ox=0, spectators=[("PPh3", 2)]),
    # optional per-species overrides:
    # dict(..., geometry={"start": "thd"}, mults={"OA": [2]}),
]

ARYL_ELECTROPHILES = {
    "PhI":   "Ic1ccccc1",
    "PhOTf": "FC(F)(F)S(=O)(=O)Oc1ccccc1",
}

CHANNELS = ["OA", "OA_cation", "XAT", "OSET"]
```

- **Library ligands**: section 3 lists molSimplify's library (about 250 entries) and checks common names.
- **SMILES ligands**: `donors` are the 1-based positions of the binding atoms among the heavy atoms
  of the SMILES string. The notebook prints them for checking; verify them, especially for ligands
  with N or P atoms that should not bind.
- **Leaving groups** (Cl, Br, I, OTf, OTs, ...) are added automatically; they do not go in `LIGANDS`.
- Each Ar–X must contain exactly one aryl C–X bond (e.g. 4-bromophenyl triflate is rejected as ambiguous).

### 3. Run sections 5–7

- **Section 5** lists every planned build: metal, oxidation state, coordination number, geometry,
  charge, multiplicities, and ligand order.
- **Section 6** runs molSimplify and shows a table. Check that `CN` equals `expected_CN`, `clashes` is
  0, and `shortest_ML` is above about 1.8 Å; rows marked `CHECK` or `FAILED` need a look (each build's
  log is in `ms_runs/<name>/molsimplify.log`).
- **Section 7** writes the inputs; **7b** shows any structure in 3D with the metal's donor atoms numbered.

## Output

Everything is written to `complex_structures/gxtb_opt/`:

| File | Contents |
|---|---|
| `<system>_<channel>[_<ArX>]_o<k>_m<mult>.com` / `.xyz` | Gaussian + xTB opt/freq input and the renumbered structure |
| `ArX_<name>`, `ArX_radanion_<name>`, `Ar_radical_<name>`, `X_anion_<X>` | Metal-free partner species (charges and multiplicities set) |
| `filenames.txt` | List of all inputs, for the SLURM array |
| `atom_index.txt` | Charge, multiplicity, key atom numbers, trans-ligand pattern, and warnings for every input |
| `g16_xtb.slurm` | SLURM array script, with `--array` already set to the number of inputs |

The naming (`_o<isomer>_m<multiplicity>`) is what [`check_gxtb_logs.py`](../check_gxtb_logs) uses to
group isomers and spin states and report their relative free energies.

To run the inputs:

```bash
cd complex_structures/gxtb_opt
sbatch g16_xtb.slurm filenames.txt
```

Edit the partition, account, and module lines in `g16_xtb.slurm` for your cluster first (see
[`slurm_g16_xtb`](../slurm_g16_xtb)).

## Tested systems

- Ni(I)/Ni(II)/Ni(III) with terpyridine (library and SMILES) and Br: PhI for all channels; PhBr,
  PhOTf, and PhOTs for the `OA` channel (PhCl and other electrophiles use the same code path)
- Pd(0)/Pd(II) with PPh₃ and PhOTf (cis and trans Pd(PPh₃)₂(Ph)(OTf))

## Known limitations

- **Bidentate ligands on three-coordinate metals** (e.g. (bpy)Ni(I)Br) built with unphysical
  metal–N distances in testing. These are flagged `CHECK`, but need to be built another way.
- **Bond lengths are generic.** With `SKIP_ANN = True`, molSimplify uses its table of metal–ligand
  distances (e.g. 2.14 Å for both Ni–N and Ni–C), regardless of oxidation state. Optimization fixes this.
- **Isomer detection** compares which donor elements are trans to each other. It cannot tell apart
  two donors of the same element in different roles, or mirror-image isomers.
- **Spin states** are counted from the metal's d electrons, assuming closed-shell ligands. Redox-active
  ligands (e.g. reduced bipyridine) are not considered.
- **`OSET` and radical-anion species** reuse the neutral geometries as starting points.
- **No conformer search.** Flexible ligands or substrates (e.g. OTs, alkyl chains) may need CREST
  after optimization.

## Citing

If you use the structures in published work, cite molSimplify as its output asks:

- E. I. Ioannidis, T. Z. H. Gani, H. J. Kulik, *J. Comput. Chem.* **2016**, 37, 2106–2117 (structure generation)

plus RDKit and, for the calculations, xTB and Gaussian 16. Check the
[molSimplify citation page](https://molsimplify.readthedocs.io/en/latest/Citation.html) for the
current list.
