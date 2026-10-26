"""
build_orthology_table.py

Builds a GENOME-WIDE one-to-one ortholog table (human as the hub species)
for mouse and zebrafish from Ensembl BioMart.

WHY: with only ~7 shared genes, most cells have zero signal and UMAP collapses
into a meaningless grid. A genome-wide shared gene space gives thousands of
informative genes to integrate and embed.

OUTPUT: orthologs_genomewide_one2one.csv with columns
    human_gene_id, human_symbol, mouse_gene_id, danio_rerio_gene_id
(plus a .provenance.txt recording which Ensembl host/release served the data).
Only genes that are one-to-one in BOTH mouse and zebrafish are kept.

HOW IT TALKS TO ENSEMBL
This script queries BioMart's web service directly with `requests` (no pybiomart cause it kept failing).
It tries several Ensembl mirrors in turn, retries transient failures, and
VALIDATES every reply before using it - an HTML error page or a shifted column
layout is rejected loudly rather than parsed into wrong orthologs.

USAGE:
    pip install requests pandas
    python build_orthology_table.py                  # normal run
    python build_orthology_table.py --diagnose       # test connectivity only
    python build_orthology_table.py --host https://useast.ensembl.org
    python build_orthology_table.py --min-confidence 1
"""

import argparse
import datetime
import io
import re
import sys
import time

import pandas as pd
import requests

# Tried in order. All serve the same BioMart service. The archive host is an
# older Ensembl release - orthology is stable across releases, but the script
# prints a warning if it ends up being used.
HOSTS = [
    "https://www.ensembl.org",
    "https://useast.ensembl.org",
    "https://asia.ensembl.org",
    "https://uswest.ensembl.org",
    "https://dec2021.archive.ensembl.org",
]
MARTSERVICE = "/biomart/martservice"
MART = "ENSEMBL_MART_ENSEMBL"
DATASET = "hsapiens_gene_ensembl"

# output column name : BioMart attribute prefix on the human dataset
TARGETS = {
    "mouse_gene_id": "mmusculus",
    "danio_rerio_gene_id": "drerio",
}
ID_PATTERN = {
    "human": re.compile(r"^ENSG\d{11}$"),
    "mmusculus": re.compile(r"^ENSMUSG\d{11}$"),
    "drerio": re.compile(r"^ENSDARG\d{11}$"),
}
HEADERS = {"User-Agent": "cross-species-immune-pipeline/1.0 (research; python-requests)"}
MIN_RAW_ROWS = 1000  # a genome-wide reply is tens of thousands of rows


# ----------------------------------------------------------------------
# Pure table logic (no network; unit-testable)
# ----------------------------------------------------------------------
def tidy_target_frame(df, prefix, out_col, min_confidence=0):
    """Reduce one raw BioMart result to unique one-to-one human->target pairs."""
    id_col = f"{prefix}_homolog_ensembl_gene"
    type_col = f"{prefix}_homolog_orthology_type"
    conf_col = f"{prefix}_homolog_orthology_confidence"

    needed = {"ensembl_gene_id", "external_gene_name", id_col, type_col}
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"result for '{prefix}' is missing columns {sorted(missing)}; "
                       f"got {list(df.columns)}")

    df = df.dropna(subset=[id_col, type_col])
    df = df[df[type_col] == "ortholog_one2one"]

    if conf_col in df.columns and min_confidence > 0:
        df = df[pd.to_numeric(df[conf_col], errors="coerce") >= min_confidence]

    # Guard against any residual many-to-one on either side
    df = df.drop_duplicates(subset=["ensembl_gene_id"], keep=False)
    df = df.drop_duplicates(subset=[id_col], keep=False)

    return df.rename(columns={id_col: out_col})[
        ["ensembl_gene_id", "external_gene_name", out_col]
    ]


def merge_targets(frames):
    """Inner-join the per-species tables on the human gene ID."""
    merged = None
    for f in frames:
        merged = f if merged is None else merged.merge(
            f.drop(columns=["external_gene_name"]), on="ensembl_gene_id", how="inner"
        )
    merged = merged.rename(
        columns={"ensembl_gene_id": "human_gene_id", "external_gene_name": "human_symbol"}
    )
    merged["human_symbol"] = merged["human_symbol"].fillna("")
    return merged.reset_index(drop=True)


# ----------------------------------------------------------------------
# BioMart access
# ----------------------------------------------------------------------
def attributes_for(prefix):
    return [
        "ensembl_gene_id",
        "external_gene_name",
        f"{prefix}_homolog_ensembl_gene",
        f"{prefix}_homolog_orthology_type",
        f"{prefix}_homolog_orthology_confidence",
    ]


def build_query_xml(attributes):
    attrs = "".join(f'<Attribute name="{a}"/>' for a in attributes)
    return (
        '<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE Query>'
        '<Query virtualSchemaName="default" formatter="TSV" header="0" '
        'uniqueRows="1" count="" datasetConfigVersion="0.6">'
        f'<Dataset name="{DATASET}" interface="default">{attrs}</Dataset></Query>'
    )


def parse_and_validate(text, prefix, attributes, min_rows=MIN_RAW_ROWS):
    """
    Turn a raw BioMart reply into a DataFrame, or raise ValueError describing
    exactly what was wrong. Columns are assigned by POSITION (header=0), so the
    content checks below are what protect against a silently shifted layout.
    """
    body = text.strip()
    if not body:
        raise ValueError("empty response from server")
    if body.startswith("<") or "Query ERROR" in body[:500]:
        raise ValueError("server returned an error/HTML page instead of data: "
                         + body[:200].replace("\n", " "))

    df = pd.read_csv(io.StringIO(text), sep="\t", header=None, dtype=str,
                     keep_default_na=True, on_bad_lines="error")
    if df.shape[1] != len(attributes):
        raise ValueError(f"expected {len(attributes)} columns, got {df.shape[1]}")
    df.columns = attributes
    if len(df) < min_rows:
        raise ValueError(f"only {len(df)} rows returned - far fewer than a genome-wide result")

    def frac_matching(series, pattern):
        s = series.dropna()
        return 0.0 if s.empty else s.map(lambda v: bool(pattern.match(v))).mean()

    checks = {
        "human gene IDs": frac_matching(df["ensembl_gene_id"], ID_PATTERN["human"]),
        f"{prefix} gene IDs": frac_matching(df[f"{prefix}_homolog_ensembl_gene"], ID_PATTERN[prefix]),
        "orthology types": df[f"{prefix}_homolog_orthology_type"].dropna()
                             .map(lambda v: v.startswith("ortholog")).mean(),
    }
    bad = {k: round(float(v), 3) for k, v in checks.items() if not v >= 0.99}
    if bad:
        raise ValueError(f"column layout does not look right (fraction valid: {bad}). "
                         f"Refusing to continue to avoid mislabelled orthologs.")
    return df


def query_host(host, prefix, timeout, retries):
    attributes = attributes_for(prefix)
    params = {"query": build_query_xml(attributes)}
    last = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(host + MARTSERVICE, params=params, headers=HEADERS,
                             timeout=timeout)
            r.raise_for_status()
            return parse_and_validate(r.text, prefix, attributes)
        except Exception as e:
            last = e
            print(f"      attempt {attempt}/{retries} failed: {type(e).__name__}: {str(e)[:220]}")
            if attempt < retries:
                time.sleep(5 * attempt)
    raise RuntimeError(f"{type(last).__name__}: {last}")


def fetch_all(hosts, min_confidence, timeout, retries):
    """Both species must come from the SAME host so the release is consistent."""
    failures = {}
    for host in hosts:
        print(f"\nTrying {host}")
        try:
            frames = []
            for out_col, prefix in TARGETS.items():
                print(f"  human -> {prefix} homologs ...")
                raw = query_host(host, prefix, timeout, retries)
                print(f"    {len(raw):,} raw rows")
                tidy = tidy_target_frame(raw, prefix, out_col, min_confidence)
                print(f"    {len(tidy):,} unique one-to-one pairs")
                frames.append(tidy)
            return merge_targets(frames), host
        except Exception as e:
            failures[host] = f"{type(e).__name__}: {str(e)[:300]}"
            print(f"  -> giving up on {host}")
    raise RuntimeError("\n".join(f"  {h}: {m}" for h, m in failures.items()))


def diagnose(hosts, timeout=30):
    """Report exactly what each host returns for the cheap 'list datasets' call."""
    print("Connectivity diagnosis (BioMart 'list datasets'):\n")
    for host in hosts:
        try:
            r = requests.get(host + MARTSERVICE, headers=HEADERS, timeout=timeout,
                             params={"type": "datasets", "mart": MART})
            has = DATASET in r.text
            print(f"{host}\n  HTTP {r.status_code} | {r.headers.get('Content-Type', '?')} | "
                  f"{len(r.text):,} chars | contains '{DATASET}': {has}")
            if not has:
                print(f"  first 250 chars: {r.text[:250]!r}")
        except Exception as e:
            print(f"{host}\n  FAILED: {type(e).__name__}: {str(e)[:250]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-confidence", type=int, default=0,
                    help="Minimum Ensembl orthology confidence (0 = no filter, 1 = high only)")
    ap.add_argument("--out", default="orthologs_genomewide_one2one.csv")
    ap.add_argument("--host", default=None, help="force one Ensembl host, e.g. https://useast.ensembl.org")
    ap.add_argument("--diagnose", action="store_true", help="test connectivity and exit")
    ap.add_argument("--timeout", type=int, default=300, help="seconds per request")
    ap.add_argument("--retries", type=int, default=2, help="attempts per query per host")
    args = ap.parse_args()

    hosts = [args.host.rstrip("/")] if args.host else HOSTS
    if args.diagnose:
        diagnose(hosts)
        return

    try:
        table, host = fetch_all(hosts, args.min_confidence, args.timeout, args.retries)
    except Exception as e:
        print("\nERROR: could not get a valid ortholog table from any host.\n")
        print(str(e))
        print("\nRunning connectivity diagnosis so the cause is visible:\n")
        diagnose(hosts)
        print("\nSend the output above (or try again later - Ensembl BioMart is "
              "sometimes down or overloaded; also try --host with another mirror, "
              "and check whether a VPN/proxy/firewall is rewriting the reply).")
        sys.exit(1)

    table.to_csv(args.out, index=False)
    note = (f"source host: {host}\nretrieved: {datetime.date.today()}\n"
            f"min_confidence: {args.min_confidence}\nrows: {len(table)}\n")
    with open(args.out.replace(".csv", ".provenance.txt"), "w") as fh:
        fh.write(note)

    print(f"\nWrote {args.out}: {len(table):,} genes one-to-one in human, mouse AND zebrafish")
    print(f"Source: {host}")
    if "archive" in host:
        print("NOTE: served by an ARCHIVE (older) Ensembl release. Orthology is stable, "
              "but record this in your methods.")
    if len(table) < 3000:
        print("WARNING: fewer than ~3,000 shared genes is unusually low - check the output.")


if __name__ == "__main__":
    main()
