"""
build_dc_umap.py  (v3.2 - DC-focused integration with a CD8+ T-cell control)

Builds one integrated cross-species UMAP from per-species .h5ad files.

WHAT CHANGED FROM v2 AND WHY
  v2 embedded EVERY cell in the GSE186158 samples. Those samples contain many
  cell types, so most of the UMAP was non-DC biology, and where the same cell
  type existed in several species the species still sat in separate blobs
  (weak batch correction). v3:
    1. ANNOTATES lineages per species (module scores on each species' own gene
       space, then cluster-level calls) -> pDC, cDC1, cDC2 and CD8_T (control).
       Markers that are not one-to-one orthologs (CLEC4C, XCR1, ...) are used
       here too, because scoring happens BEFORE reduction to the shared space.
    2. SUBSETS to DCs + CD8+ T cells (optionally plus a few 'other' cells as
       context) and BALANCES cells per species x cell type, so one species or
       one big population cannot dominate the graph.
    3. Picks HIGHLY VARIABLE GENES on raw counts (seurat_v3, per species) and
       force-includes the lineage markers that exist in the shared space.
    4. Integrates with a stronger scVI configuration (gene-batch dispersion,
       layer norm, covariates fed to the encoder), or Harmony / scANVI.
    5. Scores integration the way the reference paper does - species mixing
       (SM) and biology conservation (BC) - but the mixing null is shuffled
       WITHIN cell type, so it is not penalised for composition differences.
    6. Plots markers on a fixed 0-1 scale (clipped at a high percentile), which
       fixes the 0-4 colour ranges that differed from gene to gene.

WHAT CHANGED IN v3.1 (lineage calling only; everything else is as in v3)
  The v3 cluster calls z-scored each lineage across clusters, which hid the RAW
  size of the score. In practice that (a) labelled B-cell clusters as cDC1/cDC2
  (IRF8, IRF4, TCF4 etc. are also expressed in B cells, so a tiny DC score was a
  big outlier) and (b) missed clear CD8 T clusters, because T cells span many
  clusters so their z-scores are small. assign_cluster_labels now applies two
  raw-score rules BEFORE the old z-score logic:
    * B-cell veto : a cluster with mean B_cell score > B_VETO is never a DC.
    * CD8 rule    : T_cell score > T_MIN and CD8 score > CD8_MIN -> CD8_T.
  Tune with --b-veto / --t-min / --cd8-min, and check cluster_scores_<species>.csv.

WHAT CHANGED IN v3.2 (only the HVG clean-up changes default behaviour)
  * HVG clean-up (default ON): mitochondrial (MT-) and ribosomal (RPL/RPS) genes
    are removed from the HVG set. They are one-to-one orthologs, so they were in
    the shared space, and they carry species/technical differences rather than
    cell identity. Restore the old behaviour with --keep-mt-ribo.
  * --n-latent / --n-layers: size of the scVI model (defaults unchanged: 30 / 2).
    With only a few thousand cells a smaller model (e.g. 10 / 1) may mix better.

PIPELINE ORDER (build_orthology_table.py and load_geo_species.py are unchanged)
    python build_orthology_table.py     (once)
    python load_geo_species.py          (once; loads ALL species by Ensembl ID)
    python build_dc_umap.py --method none      <- fast diagnostic, run this first
    python build_dc_umap.py --method harmony
    python build_dc_umap.py --method scvi      <- the real integration (slower)
    python build_dc_umap.py --method scanvi    <- semi-supervised (see note below)

  Check  cluster_scores_<species>.csv  after the first run: the lineage calls
  are marker-based and thresholds may need tuning (--z-thr, --annot-resolution)
  or manual fixes (--overrides overrides.csv with columns species,cluster,cell_type).

Expects per-species .h5ad files whose var_names are Ensembl gene IDs and whose
var has a 'gene_symbols' column (load_geo_species.py produces exactly this).
Extra dependencies:  pip install scikit-misc igraph scikit-learn harmonypy
"""

import argparse
import os
import re

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

SPECIES_FILES = {
    "human": "h5ad_files/human.h5ad",
    "mouse": "h5ad_files/mouse.h5ad",
    "danio_rerio": "h5ad_files/danio_rerio.h5ad",
}

ORTHOLOG_TABLE = "orthologs_genomewide_one2one.csv"

# The curated human DC markers (pDC / cDC1 / cDC2) plus the CD8 T-cell control
# markers. Used only as an overlay on the final UMAP (only those that are
# one-to-one in all three species can appear there).
MARKER_SYMBOLS = [
    "ITGAX", "CLEC4C", "NRP1", "LILRA4", "IRF7", "LILRB4", "TCF4", "IRF8",
    "THBD", "XCR1", "ID2", "CD1C", "ZBTB46", "IRF4", "CLEC10A", "CD14",
    "CD3E", "CD8A", "CD8B",
]

# Marker sets used to CALL lineages. Matching is by gene symbol, case-insensitive,
# in each species' own gene space (zebrafish paralogs such as id2a/id2b/xcr1a.1
# are matched too). Symbols a species does not have are simply ignored.
# Tune these lists freely - they are the main knob for annotation quality.
LINEAGE_MARKERS = {
    "pDC":     ["CLEC4C", "LILRA4", "IL3RA", "NRP1", "IRF7", "LILRB4", "TCF4",
                "SIGLECH", "LY6D", "CCR9"],
    "cDC1":    ["XCR1", "CLEC9A", "BATF3", "THBD", "IRF8"],
    "cDC2":    ["CD1C", "CLEC10A", "FCER1A", "IRF4", "SIRPA", "ZBTB46"],
    "T_cell":  ["CD3E", "CD3D", "CD3G", "LCK", "ZAP70", "CD2"],
    "CD8":     ["CD8A", "CD8B"],
    # competitors: if one of these wins a cluster, it is NOT called a DC
    "B_cell":  ["CD79A", "CD79B", "MS4A1", "PAX5"],
    "Myeloid": ["CSF1R", "CD68", "LYZ", "S100A8", "S100A9"],
}
DC_LABELS = ["pDC", "cDC1", "cDC2"]

# Raw-score rules used by assign_cluster_labels (module scores, cluster means).
# Override from the command line with --b-veto / --t-min / --cd8-min.
B_VETO = 0.3     # B_cell score above this: cluster is never called a DC
T_MIN = 0.3      # T_cell score needed for the raw CD8 rule
CD8_MIN = 0.8    # CD8 score needed for the raw CD8 rule
FINAL_LABELS = DC_LABELS + ["CD8_T"]      # CD8_T = the control population

# v3.2 options (set from the command line in __main__)
N_LATENT, N_LAYERS = 30, 2                 # --n-latent / --n-layers
EXCLUDE_MT_RIBO = True                     # --keep-mt-ribo turns this off

CELL_TYPE_COLORS = {
    "pDC": "#1f77b4", "cDC1": "#d62728", "cDC2": "#2ca02c",
    "CD8_T": "#ff7f0e", "other": "#bdbdbd",
}
SPECIES_COLORS = {"human": "#e6550d", "mouse": "#2ca02c", "danio_rerio": "#3182bd"}


# ----------------------------------------------------------------------
# Orthology
# ----------------------------------------------------------------------
def load_id_maps(path):
    """
    {species: {species_gene_id: human_gene_id}} plus {human_gene_id: symbol}.
    Human is its own hub, so its map is the identity on the table's genes.
    """
    t = pd.read_csv(path)
    t = t.dropna(subset=["human_gene_id", "mouse_gene_id", "danio_rerio_gene_id"])
    maps = {
        "human": {g: g for g in t["human_gene_id"]},
        "mouse": dict(zip(t["mouse_gene_id"], t["human_gene_id"])),
        "danio_rerio": dict(zip(t["danio_rerio_gene_id"], t["human_gene_id"])),
    }
    symbols = dict(zip(t["human_gene_id"], t["human_symbol"].fillna("")))
    print(f"Ortholog table: {len(t):,} genes one-to-one in human, mouse and zebrafish")
    return maps, symbols


# ----------------------------------------------------------------------
# Lineage annotation (per species, in the species' OWN gene space)
# ----------------------------------------------------------------------
def resolve_markers(var_symbols, wanted, species):
    """
    Map wanted (human-style) symbols to var_names of this species.
    Exact case-insensitive match; for non-human species also accept a paralog
    suffix (id2 -> id2a / id2b, xcr1 -> xcr1a.1).
    """
    sym = var_symbols.astype(str)
    up = sym.str.upper()
    found = []
    for w in wanted:
        hit = up == w
        if not hit.any() and species != "human":
            hit = up.str.match(rf"^{re.escape(w)}[A-Z]?(\.\d+)?$")
        found.extend(sym.index[hit.values].tolist())
    return list(dict.fromkeys(found))


def score_lineages(adata, species):
    """Add obs['score_<lineage>'] (Scanpy module scores) for every lineage."""
    if "gene_symbols" in adata.var.columns:
        symbols = adata.var["gene_symbols"]
    else:
        symbols = pd.Series(adata.var_names, index=adata.var_names)
    for lineage, wanted in LINEAGE_MARKERS.items():
        genes = resolve_markers(symbols, wanted, species)
        col = f"score_{lineage}"
        if not genes:
            adata.obs[col] = np.nan
            print(f"    {lineage:<8} 0 marker genes found in {species}")
            continue
        sc.tl.score_genes(adata, gene_list=genes, score_name=col, use_raw=False)
        names = sorted(set(symbols.loc[genes].astype(str)))
        print(f"    {lineage:<8} {len(genes)} genes: {', '.join(names)}")


def cluster_species(adata, resolution, n_hvg=2000, n_pcs=30):
    """Quick per-species clustering used only for lineage calling."""
    hv = sc.pp.highly_variable_genes(
        adata, n_top_genes=min(n_hvg, adata.n_vars - 1), flavor="seurat", inplace=False
    )
    tmp = adata[:, hv["highly_variable"].values].copy()
    sc.pp.scale(tmp, max_value=10)
    sc.pp.pca(tmp, n_comps=min(n_pcs, tmp.n_vars - 1, tmp.n_obs - 1))
    sc.pp.neighbors(tmp, n_neighbors=15)
    sc.tl.leiden(tmp, resolution=resolution, flavor="igraph",
                 n_iterations=2, directed=False, key_added="ann_cluster")
    return tmp.obs["ann_cluster"].astype(str)


def assign_cluster_labels(cluster_scores, z_thr, min_raw=0.0):
    """
    cluster_scores: clusters x lineages (mean module score).
    Raw-score rules come first (v3.1):
      1. B-cell veto: B_cell score > B_VETO -> 'other' (B cells share IRF8 /
         IRF4 / TCF4 with DCs and were being called cDC1 / cDC2).
      2. CD8 rule: T_cell > T_MIN and CD8 > CD8_MIN -> 'CD8_T'.
    Otherwise a cluster takes the lineage whose score is highest RELATIVE to the
    other clusters (z-score across clusters), if it clears z_thr and a positive
    raw score. T-cell clusters become CD8_T only if the CD8 score also clears
    z_thr. Competitor lineages (B_cell, Myeloid) and anything weak become 'other'.
    """
    cs = cluster_scores
    z = (cs - cs.mean()) / cs.std(ddof=0).replace(0, np.nan)
    z = z.fillna(-np.inf)
    contenders = ["pDC", "cDC1", "cDC2", "T_cell", "B_cell", "Myeloid"]
    labels = {}
    for cl in cs.index:
        r, zr = cs.loc[cl], z.loc[cl]
        if r["B_cell"] > B_VETO:
            labels[cl] = "other"
            continue
        if r["T_cell"] > T_MIN and r["CD8"] > CD8_MIN:
            labels[cl] = "CD8_T"
            continue
        top = zr[contenders].idxmax()
        if not np.isfinite(zr[top]) or zr[top] < z_thr or not r[top] > min_raw:
            labels[cl] = "other"
        elif top == "T_cell":
            ok = zr["CD8"] >= z_thr and r["CD8"] > min_raw
            labels[cl] = "CD8_T" if ok else "other"
        elif top in DC_LABELS:
            labels[cl] = top
        else:
            labels[cl] = "other"
    return labels


def annotate_lineages(adata, species, resolution, z_thr, overrides=None):
    """Adds obs['cell_type'] and the score_* columns; writes cluster_scores_<species>.csv."""
    print(f"  scoring lineage markers ({species}):")
    score_lineages(adata, species)
    adata.obs["ann_cluster"] = cluster_species(adata, resolution)

    cols = [f"score_{l}" for l in LINEAGE_MARKERS]
    cs = adata.obs.groupby("ann_cluster", observed=True)[cols].mean()
    cs.columns = list(LINEAGE_MARKERS)
    labels = assign_cluster_labels(cs, z_thr)
    if overrides is not None:
        for _, r in overrides[overrides["species"] == species].iterrows():
            labels[str(r["cluster"])] = r["cell_type"]
    cs["n_cells"] = adata.obs["ann_cluster"].value_counts()
    cs["cell_type"] = pd.Series(labels)
    cs.round(3).to_csv(f"cluster_scores_{species}.csv")

    adata.obs["cell_type"] = adata.obs["ann_cluster"].map(labels).astype(str)
    counts = adata.obs["cell_type"].value_counts().to_dict()
    print(f"  lineage calls ({species}): {counts}  "
          f"-> review cluster_scores_{species}.csv")
    return adata


# ----------------------------------------------------------------------
# Per-species preprocessing
# ----------------------------------------------------------------------
def preprocess_species(adata, species, id_map, annot_resolution=2.0,
                       z_thr=1.5, overrides=None):
    adata = adata.copy()
    # strip Ensembl version suffixes (ENSG...1.5 -> ENSG...1) so IDs match the table
    adata.var_names = adata.var_names.str.replace(r"\.\d+$", "", regex=True)
    adata.var_names_make_unique()
    n_start = adata.n_obs

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    # Mitochondrial genes are identified from SYMBOLS, not var_names (which
    # are Ensembl IDs). Works for human 'MT-', mouse/zebrafish 'mt-' via upper().
    if "gene_symbols" in adata.var.columns:
        sym = adata.var["gene_symbols"].astype(str)
    else:
        sym = pd.Series(adata.var_names, index=adata.var_names)
    adata.var["mt"] = sym.str.upper().str.startswith("MT-").values
    n_mt = int(adata.var["mt"].sum())
    if n_mt == 0:
        print(f"  WARNING [{species}]: no mitochondrial genes found - "
              f"mito QC filter cannot be applied. Check gene_symbols column.")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, percent_top=None)
    adata = adata[adata.obs["pct_counts_mt"] < 20].copy()

    # scVI needs RAW counts: stash them before normalization
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # NEW in v3: call lineages while ALL genes are still present
    adata = annotate_lineages(adata, species, annot_resolution, z_thr, overrides)

    # Subset to one-to-one orthologs and rename onto the shared human gene ID
    keep = adata.var_names.isin(id_map.keys())
    n_genes_before = adata.n_vars
    adata = adata[:, keep].copy()
    adata.var_names = [id_map[g] for g in adata.var_names]
    assert adata.var_names.is_unique, f"{species}: duplicate shared IDs after renaming"

    adata.obs["species"] = species
    print(f"  {species}: {n_start:,} -> {adata.n_obs:,} cells after QC; "
          f"{n_mt} mito genes found; {adata.n_vars:,}/{n_genes_before:,} genes "
          f"map to the shared ortholog space")
    return adata


def select_and_balance(adata, keep_other, max_per_group, seed=0):
    """Keep DCs + CD8 T (+ optional 'other' context) and cap cells per species x type."""
    rng = np.random.default_rng(seed)
    ct = adata.obs["cell_type"].astype(str).values
    sp_ = adata.obs["species"].astype(str).values
    keep = []
    for s in np.unique(sp_):
        for t in FINAL_LABELS + ["other"]:
            idx = np.where((sp_ == s) & (ct == t))[0]
            cap = keep_other if t == "other" else max_per_group
            if len(idx) > cap:
                idx = rng.choice(idx, cap, replace=False)
            keep.extend(idx.tolist())
    out = adata[np.sort(keep)].copy()

    tab = pd.crosstab(out.obs["species"], out.obs["cell_type"])
    print("\nCells kept (species x cell type):")
    print(tab.to_string())
    for s in tab.index:
        for t in FINAL_LABELS:
            n = tab.loc[s, t] if t in tab.columns else 0
            if n < 30:
                print(f"  NOTE: only {n} {t} cells in {s} - interpret that cell "
                      f"type in this species with caution (or tune annotation).")
    out.obs["cell_type"] = pd.Categorical(
        out.obs["cell_type"], categories=[c for c in CELL_TYPE_COLORS if c in set(out.obs["cell_type"])]
    )
    return out


# ----------------------------------------------------------------------
# Quantitative diagnostics: species mixing (SM) and biology conservation (BC)
# ----------------------------------------------------------------------
def _knn_entropy(distances, codes, n_cat):
    """Per-cell normalized entropy of species composition among its kNN."""
    D = distances.tocsr()
    out = np.zeros(D.shape[0])
    for i in range(D.shape[0]):
        nbrs = D.indices[D.indptr[i]:D.indptr[i + 1]]
        if len(nbrs) == 0:
            continue
        p = np.bincount(codes[nbrs], minlength=n_cat).astype(float)
        p = p[p > 0] / p.sum()
        out[i] = -(p * np.log(p)).sum() / np.log(n_cat)
    return out


def _permute_within(codes, groups, rng):
    out = codes.copy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        out[idx] = rng.permutation(codes[idx])
    return out


def report_species_mixing(adata, key="species", group_key="cell_type", seed=0):
    """
    Mean kNN species-entropy (0 = every neighbourhood is one species,
    1 = perfectly even), compared with the same statistic after shuffling
    species labels WITHIN each cell type. A well-integrated embedding
    approaches the shuffled value. Shuffling within cell type means a cell
    type that is missing from a species is not counted as a mixing failure.
    """
    cats = adata.obs[key].astype("category")
    codes = cats.cat.codes.values
    n_cat = len(cats.cat.categories)
    groups = adata.obs[group_key].astype(str).values
    D = adata.obsp["distances"]

    obs_ent = _knn_entropy(D, codes, n_cat)
    rng = np.random.default_rng(seed)
    null_ent = _knn_entropy(D, _permute_within(codes, groups, rng), n_cat)

    print("\nSpecies mixing (kNN entropy, 0 = unmixed, 1 = even):")
    print(f"  observed overall : {obs_ent.mean():.3f}")
    print(f"  shuffled baseline: {null_ent.mean():.3f}   <- target (shuffled within cell type)")
    print(f"  SM score (observed / baseline): {obs_ent.mean() / max(null_ent.mean(), 1e-9):.3f}")
    for g in pd.unique(groups):
        m = groups == g
        print(f"  {g:<8} observed {obs_ent[m].mean():.3f} | baseline {null_ent[m].mean():.3f}")
    adata.obs["species_mixing_entropy"] = obs_ent
    return obs_ent.mean(), null_ent.mean()


def report_celltype_separation(adata, rep, key="cell_type", n=5000, seed=0):
    """
    BC proxy: silhouette of cell-type labels in the integration space, rescaled
    to 0-1. High = pDC / cDC1 / cDC2 / CD8_T stay distinct after integration.
    """
    from sklearn.metrics import silhouette_score
    labels = adata.obs[key].astype(str).values
    if len(set(labels)) < 2:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.choice(adata.n_obs, min(n, adata.n_obs), replace=False)
    s = silhouette_score(adata.obsm[rep][idx], labels[idx])
    bc = (s + 1) / 2
    print(f"\nBiology conservation proxy (cell-type silhouette, 0-1): {bc:.3f}")
    return bc


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------
def select_hvgs(adata, n_hvg, force_symbols):
    """HVGs from raw counts, ranked across species; lineage markers are force-kept."""
    sym = adata.var["symbol"].astype(str).str.upper()
    excl = (sym.str.startswith("MT-") | sym.str.match(r"^RP[SL]\d")).values \
        if EXCLUDE_MT_RIBO else np.zeros(adata.n_vars, dtype=bool)
    n = min(n_hvg + int(excl.sum()), adata.n_vars - 1)   # over-select, then drop
    try:
        sc.pp.highly_variable_genes(adata, n_top_genes=n, flavor="seurat_v3",
                                    layer="counts", batch_key="species")
    except Exception as e:  # e.g. scikit-misc missing
        print(f"(seurat_v3 HVGs failed: {type(e).__name__}: {e}; "
              f"falling back to flavor='seurat'. Optional: pip install scikit-misc)")
        sc.pp.highly_variable_genes(adata, n_top_genes=n, flavor="seurat",
                                    batch_key="species")
    force = adata.var["symbol"].astype(str).str.upper().isin(force_symbols).values
    adata.var["highly_variable"] = (adata.var["highly_variable"].values & ~excl) | force
    print(f"HVGs: {int(adata.var['highly_variable'].sum()):,} "
          f"(incl. {int(force.sum())} forced lineage/overlay markers; "
          f"{int(excl.sum())} MT-/ribosomal genes excluded)")


def build_integrated_umap(species_files, ortholog_table, method="scvi",
                          n_hvg=3000, scvi_epochs=None, annot_resolution=2.0,
                          z_thr=1.5, overrides=None, keep_other=0,
                          max_per_group=3000, n_neighbors=30, min_dist=0.3,
                          harmony_theta=2.0):
    maps, symbols = load_id_maps(ortholog_table)

    adatas = []
    for species, path in species_files.items():
        print(f"\nLoading {species} from {path} ...")
        if species not in maps:
            raise ValueError(f"No ortholog map for '{species}'. "
                             f"Known: {list(maps)}")
        adatas.append(preprocess_species(
            sc.read_h5ad(path), species, maps[species],
            annot_resolution=annot_resolution, z_thr=z_thr, overrides=overrides))

    print("\nConcatenating species ...")
    adata_all = ad.concat(
        adatas, join="inner", label="species_batch",
        keys=[a.obs["species"].iloc[0] for a in adatas], index_unique="-",
    )
    print(f"Combined: {adata_all.n_obs:,} cells x {adata_all.n_vars:,} shared genes")
    if adata_all.n_vars < 500:
        raise RuntimeError(
            f"Only {adata_all.n_vars} shared genes. The gene IDs in the .h5ad "
            f"files probably do not match the ortholog table (re-run "
            f"load_geo_species.py so all species load by Ensembl ID)."
        )
    adata_all.var["symbol"] = [symbols.get(g, g) for g in adata_all.var_names]

    # DCs + CD8 T control, balanced across species
    adata_all = select_and_balance(adata_all, keep_other, max_per_group)

    # Highly variable genes (recomputed on the DC/T subset so they separate THESE cells)
    select_hvgs(adata_all, n_hvg, set(MARKER_SYMBOLS))
    adata_hvg = adata_all[:, adata_all.var["highly_variable"].values].copy()
    print(f"Using {adata_hvg.n_vars:,} genes for the embedding")

    if method in ("none", "harmony"):
        sc.pp.scale(adata_hvg, max_value=10)
        sc.pp.pca(adata_hvg, n_comps=30)
        rep = "X_pca"
        if method == "harmony":
            # harmonypy called directly: sc.external's wrapper breaks on some
            # harmonypy versions (returns the matrix transposed)
            import harmonypy
            ho = harmonypy.run_harmony(adata_hvg.obsm["X_pca"], adata_hvg.obs,
                                       "species", theta=harmony_theta)
            Z = np.asarray(ho.Z_corr)
            adata_hvg.obsm["X_pca_harmony"] = Z if Z.shape[0] == adata_hvg.n_obs else Z.T
            rep = "X_pca_harmony"
    elif method in ("scvi", "scanvi"):
        import scvi
        scvi.model.SCVI.setup_anndata(adata_hvg, layer="counts", batch_key="species")
        model = scvi.model.SCVI(
            adata_hvg, n_latent=N_LATENT, n_layers=N_LAYERS, gene_likelihood="nb",
            dispersion="gene-batch",          # species differ in noise per gene
            use_layer_norm="both", use_batch_norm="none",
            encode_covariates=True,           # encoder sees species -> stronger correction
        )
        model.train(max_epochs=scvi_epochs, early_stopping=True)
        rep = "X_scVI"
        adata_hvg.obsm[rep] = model.get_latent_representation()
        if method == "scanvi":
            # Semi-supervised: uses the marker-derived labels. Cell-type structure
            # is then partly imposed, so report scANVI results as such and use
            # 'scvi' / 'harmony' for any claim about conservation of DC identity.
            # Labels go in as PLAIN STRINGS (not a pandas Categorical). A categorical
            # that lists 'Unknown' when no cell carries it makes scvi's label codes and
            # classifier size disagree ("Target N is out of bounds"). With plain strings
            # scvi builds the mapping from the values actually present and appends
            # 'Unknown' itself as the unlabeled class.
            lab = adata_hvg.obs["cell_type"].astype(str)
            adata_hvg.obs["cell_type_scanvi"] = (
                lab.where(lab != "other", "Unknown").astype(str).values)
            print("scANVI labels:",
                  adata_hvg.obs["cell_type_scanvi"].value_counts().to_dict())
            lvae = scvi.model.SCANVI.from_scvi_model(
                model, labels_key="cell_type_scanvi", unlabeled_category="Unknown")
            lvae.train(max_epochs=20, n_samples_per_label=100)
            rep = "X_scANVI"
            adata_hvg.obsm[rep] = lvae.get_latent_representation()
    else:
        raise ValueError("method must be 'none', 'harmony', 'scvi' or 'scanvi'")

    adata_all.obsm[rep] = adata_hvg.obsm[rep]
    sc.pp.neighbors(adata_all, use_rep=rep, n_neighbors=n_neighbors)
    sc.tl.umap(adata_all, min_dist=min_dist)

    try:  # optional: clusters for downstream work
        sc.tl.leiden(adata_all, resolution=0.5, flavor="igraph",
                     n_iterations=2, directed=False, key_added="leiden")
    except Exception as e:
        print(f"(skipping Leiden clustering: {type(e).__name__}: {e}. "
              f"Optional: pip install igraph)")

    report_species_mixing(adata_all)
    report_celltype_separation(adata_all, rep)
    return adata_all


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def add_marker_expression(adata, symbols, clip_pct=99, smooth=False):
    """
    Copy each marker's expression into adata.obs for plotting, optionally
    kNN-smoothed (reduces dropout speckle), clipped at the clip_pct percentile
    of expressing cells and rescaled to 0-1, so every panel shares one range.
    """
    first_idx = {}
    for i, s in enumerate(adata.var["symbol"].astype(str)):
        first_idx.setdefault(s.upper(), i)
    C = adata.obsp["connectivities"] if smooth and "connectivities" in adata.obsp else None
    present, missing = [], []
    for m in symbols:
        if m not in first_idx:
            missing.append(m)
            continue
        col = adata.X[:, first_idx[m]]
        col = col.toarray().ravel() if sp.issparse(col) else np.asarray(col).ravel()
        if C is not None:
            col = (C @ col + col) / (np.asarray(C.sum(axis=1)).ravel() + 1.0)
        nz = col[col > 0]
        if nz.size == 0:
            missing.append(m)
            continue
        vmax = np.percentile(nz, clip_pct)
        adata.obs[f"expr_{m}"] = np.clip(col, 0, vmax) / vmax
        present.append(m)
    return present, missing


def plot_split_by_species(adata, color, path):
    """One panel per species (all cells grey), like Fig. 4a of the reference paper."""
    species = list(adata.obs["species"].astype(str).unique())
    fig, axes = plt.subplots(1, len(species), figsize=(4.8 * len(species), 4.2))
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, species):
        sc.pl.umap(adata, ax=ax, show=False, frameon=False, size=6)
        sub = adata[adata.obs["species"].astype(str) == s]
        sc.pl.umap(sub, color=color, palette=CELL_TYPE_COLORS, ax=ax, show=False,
                   frameon=False, size=10, title=s,
                   legend_loc="right margin" if ax is axes[-1] else None)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_results(adata, out_prefix="dc_umap", clip_pct=99, smooth=False):
    os.makedirs("figures", exist_ok=True)
    present, missing = add_marker_expression(adata, MARKER_SYMBOLS, clip_pct, smooth)
    print(f"\nMarkers in the shared gene space ({len(present)}/{len(MARKER_SYMBOLS)}): {present}")
    if missing:
        print(f"Not one-to-one in all three species (or never expressed), so absent: {missing}")

    sc.pl.umap(adata, color="species", palette=SPECIES_COLORS, size=8,
               save=f"_{out_prefix}_species.png", show=False)
    sc.pl.umap(adata, color="cell_type", palette=CELL_TYPE_COLORS, size=8,
               save=f"_{out_prefix}_celltype.png", show=False)
    plot_split_by_species(adata, "cell_type", f"figures/umap_{out_prefix}_celltype_by_species.png")

    if present:  # fixed 0-1 colour range, expressing cells drawn on top
        sc.pl.umap(adata, color=[f"expr_{m}" for m in present], title=present,
                   ncols=4, vmin=0, vmax=1, cmap="viridis", sort_order=True,
                   size=8, save=f"_{out_prefix}_markers.png", show=False)

    # lineage module scores: robust signature view (uses non-one-to-one markers too)
    sc.pl.umap(adata, color=[f"score_{l}" for l in ["pDC", "cDC1", "cDC2", "T_cell", "CD8"]
                             if adata.obs[f"score_{l}"].notna().any()],
               vmin=0, vmax="p99", cmap="viridis", sort_order=True, ncols=3, size=8,
               save=f"_{out_prefix}_lineage_scores.png", show=False)

    if "leiden" in adata.obs:
        sc.pl.umap(adata, color="leiden", legend_loc="on data",
                   save=f"_{out_prefix}_leiden.png", show=False)
    print(f"Plots written to figures/umap_{out_prefix}_*.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["none", "harmony", "scvi", "scanvi"], default="scvi",
                    help="'none' = PCA only (fast diagnostic, expect species to separate); "
                         "'harmony' / 'scvi' = batch-corrected; "
                         "'scanvi' = scVI + marker-derived labels (semi-supervised)")
    ap.add_argument("--orthologs", default=ORTHOLOG_TABLE)
    ap.add_argument("--n-hvg", type=int, default=3000)
    ap.add_argument("--scvi-epochs", type=int, default=None,
                    help="override scVI epochs (default: scVI's own heuristic)")
    ap.add_argument("--annot-resolution", type=float, default=2.0,
                    help="Leiden resolution of the per-species clustering used for lineage calls")
    ap.add_argument("--z-thr", type=float, default=1.5,
                    help="lineage score z-score (across clusters) needed for a call; lower = more permissive")
    ap.add_argument("--b-veto", type=float, default=B_VETO,
                    help="cluster B_cell score above this is never called a DC")
    ap.add_argument("--t-min", type=float, default=T_MIN,
                    help="T_cell score needed (with --cd8-min) to call a cluster CD8_T by raw score")
    ap.add_argument("--cd8-min", type=float, default=CD8_MIN,
                    help="CD8 score needed (with --t-min) to call a cluster CD8_T by raw score")
    ap.add_argument("--overrides", default=None,
                    help="CSV with columns species,cluster,cell_type to hand-correct lineage calls")
    ap.add_argument("--keep-other", type=int, default=0,
                    help="also keep this many 'other' cells per species as grey context")
    ap.add_argument("--max-per-group", type=int, default=3000,
                    help="cap on cells per species x cell type (balances the graph)")
    ap.add_argument("--n-latent", type=int, default=N_LATENT, help="scVI latent dimensions")
    ap.add_argument("--n-layers", type=int, default=N_LAYERS, help="scVI hidden layers")
    ap.add_argument("--keep-mt-ribo", action="store_true",
                    help="do NOT remove MT-/ribosomal genes from the HVG set (v3.1 behaviour)")
    ap.add_argument("--n-neighbors", type=int, default=30)
    ap.add_argument("--min-dist", type=float, default=0.3)
    ap.add_argument("--marker-clip-pct", type=float, default=99,
                    help="marker colour scale is clipped here and rescaled to 0-1")
    ap.add_argument("--smooth-markers", action="store_true",
                    help="kNN-smooth marker expression before plotting")
    ap.add_argument("--out", default="dc_umap_genomewide.h5ad")
    args = ap.parse_args()
    B_VETO, T_MIN, CD8_MIN = args.b_veto, args.t_min, args.cd8_min
    N_LATENT, N_LAYERS = args.n_latent, args.n_layers
    EXCLUDE_MT_RIBO = not args.keep_mt_ribo

    overrides = pd.read_csv(args.overrides, dtype=str) if args.overrides else None
    adata_all = build_integrated_umap(
        SPECIES_FILES, args.orthologs, method=args.method, n_hvg=args.n_hvg,
        scvi_epochs=args.scvi_epochs, annot_resolution=args.annot_resolution,
        z_thr=args.z_thr, overrides=overrides, keep_other=args.keep_other,
        max_per_group=args.max_per_group, n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
    )
    plot_results(adata_all, out_prefix=f"dc_{args.method}",
                 clip_pct=args.marker_clip_pct, smooth=args.smooth_markers)
    adata_all.write(args.out)
    print(f"Saved {args.out}")
