"""Disk-backed blocking index and high-recall multi-rule candidate retrieval."""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Iterable

from rapidfuzz import fuzz, process

from . import config
from .normalization import compact_name, normalize_address, normalize_country, normalize_name, text_tokens

_RECORD_COLUMNS = ("entity_id", "source", "country", "name_norm", "name_compact", "name_prefix", "address_norm")
_EXACT_BLOCK_SQL = " UNION ALL ".join(f"SELECT * FROM ({branch})" for branch in [
    "SELECT rowid,entity_id,source,country,name_norm,name_compact,name_prefix,address_norm,'name_exact' AS rule FROM records WHERE source=? AND country=? AND name_norm=? LIMIT ?",
    "SELECT rowid,entity_id,source,country,name_norm,name_compact,name_prefix,address_norm,'name_compact' AS rule FROM records WHERE source=? AND country=? AND name_compact=? LIMIT ?",
    "SELECT rowid,entity_id,source,country,name_norm,name_compact,name_prefix,address_norm,'name_prefix' AS rule FROM records WHERE source=? AND country=? AND name_prefix=? LIMIT ?",
    "SELECT rowid,entity_id,source,country,name_norm,name_compact,name_prefix,address_norm,'address_exact' AS rule FROM records WHERE source=? AND country=? AND address_norm=? LIMIT ?",
])


def connect_index(path: str | Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(path), timeout=120)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout=120000")
    con.execute("PRAGMA cache_size=-262144")  # 256 MiB page cache for repeated block lookups
    con.execute("PRAGMA temp_store=FILE")
    return con


def build_index(source_files: dict[str, str | Path], output_path: str | Path) -> Path:
    """Index Source 2/3 rows and FTS terms without loading either file in memory."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    con = connect_index(output_path)
    try:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS records (
                rowid INTEGER PRIMARY KEY,
                entity_id TEXT NOT NULL UNIQUE,
                source TEXT NOT NULL,
                country TEXT NOT NULL,
                name_norm TEXT NOT NULL,
                name_compact TEXT NOT NULL,
                name_prefix TEXT NOT NULL,
                address_norm TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_s2 USING fts5(name_norm, address_norm, tokenize='unicode61 remove_diacritics 2');
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_s3 USING fts5(name_norm, address_norm, tokenize='unicode61 remove_diacritics 2');
        """)
        file_signature = {key: {"path": str(Path(path).resolve()), "size": Path(path).stat().st_size,
                                "mtime_ns": Path(path).stat().st_mtime_ns}
                          for key, path in source_files.items()}
        stored = con.execute("SELECT value FROM metadata WHERE key='signature'").fetchone()
        signature = json.dumps(file_signature, sort_keys=True)
        if stored and stored[0] == signature and con.execute("SELECT count(*) FROM records").fetchone()[0] > 0:
            return output_path
        con.execute("DELETE FROM metadata")
        con.execute("DELETE FROM records")
        con.execute("DELETE FROM fts_s2")
        con.execute("DELETE FROM fts_s3")
        con.commit()
        next_rowid = 1
        batch_records = []
        for source, path in source_files.items():
            expected = "S2" if source.endswith("2") else "S3"
            with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream, delimiter="\t")
                if not set(config.SOURCE_COLUMNS).issubset(reader.fieldnames or []):
                    raise ValueError(f"Unexpected schema in {path}")
                fts_name = "fts_s2" if expected == "S2" else "fts_s3"
                batch_fts = []
                for row in reader:
                    entity_id = row["entity_id"].strip()
                    if not entity_id.startswith(expected + "-"):
                        raise ValueError(f"Unexpected ID prefix: {entity_id}")
                    name_norm = normalize_name(row["business_name"])
                    address_norm = normalize_address(row["business_address"])
                    batch_records.append((next_rowid, entity_id, expected,
                                          normalize_country(row["country"]), name_norm,
                                          name_norm.replace(" ", ""), name_norm[:5], address_norm))
                    batch_fts.append((next_rowid, name_norm, address_norm))
                    next_rowid += 1
                    if len(batch_records) >= config.SQLITE_BATCH_SIZE:
                        con.executemany("INSERT INTO records VALUES (?,?,?,?,?,?,?,?)", batch_records)
                        con.executemany(f"INSERT INTO {fts_name}(rowid,name_norm,address_norm) VALUES (?,?,?)", batch_fts)
                        con.commit()
                        batch_records.clear(); batch_fts.clear()
                if batch_records:
                    con.executemany("INSERT INTO records VALUES (?,?,?,?,?,?,?,?)", batch_records)
                    con.executemany(f"INSERT INTO {fts_name}(rowid,name_norm,address_norm) VALUES (?,?,?)", batch_fts)
                    con.commit()
                    batch_records.clear()
        con.executescript("""
            CREATE INDEX IF NOT EXISTS idx_records_name ON records(source,country,name_norm);
            CREATE INDEX IF NOT EXISTS idx_records_compact ON records(source,country,name_compact);
            CREATE INDEX IF NOT EXISTS idx_records_prefix ON records(source,country,name_prefix);
            CREATE INDEX IF NOT EXISTS idx_records_address ON records(source,country,address_norm);
        """)
        con.execute("INSERT OR REPLACE INTO metadata VALUES ('signature',?)", (signature,))
        con.commit()
        return output_path
    finally:
        con.close()


def _fts_field_candidates(con: sqlite3.Connection, source: str, field: str, text: str,
                          country: str, top_k: int, rule: str,
                          candidates: dict[int, dict]) -> None:
    fts_table = "fts_s2" if source == "S2" else "fts_s3"
    tokens = set(text_tokens(text, min_length=3))
    if field == "address_norm":
        numeric = sorted((t for t in tokens if any(ch.isdigit() for ch in t)), key=len, reverse=True)
        lexical = sorted((t for t in tokens if len(t) >= 5 and not any(ch.isdigit() for ch in t)),
                         key=len, reverse=True)
        chosen = (numeric + lexical)[:3]
    else:
        chosen = sorted((t for t in tokens if len(t) >= 5), key=len, reverse=True)[:3]
    if not chosen:
        return
    quoted = ['"' + t.replace('"', '""') + '"' for t in chosen]
    queries = [f'{field} : (' + " OR ".join(quoted) + ")"]
    if len(quoted) >= 2:
        queries.append(f'{field} : (' + " AND ".join(quoted[:2]) + ")")
    sql = (f"SELECT r.rowid,r.entity_id,r.source,r.country,r.name_norm,r.address_norm "
           f"FROM {fts_table} f JOIN records r ON r.rowid=f.rowid "
           f"WHERE {fts_table} MATCH ? AND r.country=? LIMIT ?")
    pool_by_id = {}
    for query in queries:
        for row in con.execute(sql, (query, country, config.BLOCK_FTS_POOL_SIZE)):
            pool_by_id[int(row["rowid"])] = row
    pool = list(pool_by_id.values())
    if not pool:
        return
    ranked = process.extract(text, [row[field] for row in pool],
                             scorer=fuzz.token_set_ratio, limit=top_k)
    for _, _, index in ranked:
        row = pool[index]
        rid = int(row["rowid"])
        if rid not in candidates:
            candidates[rid] = {**dict(row), "rules": set()}
        candidates[rid]["rules"].add(rule)


def generate_candidates(con: sqlite3.Connection, business_name: str,
                        business_address: str, country: str) -> list[dict]:
    """Union exact, compact, prefix, address, and FTS name/address blocks."""
    name = normalize_name(business_name)
    compact = compact_name(business_name)
    address = normalize_address(business_address)
    country_norm = normalize_country(country)
    if not country_norm:
        return []
    candidates: dict[int, dict] = {}
    for source in ("S2", "S3"):
        params = (source, country_norm, name, config.BLOCK_EXACT_LIMIT,
                  source, country_norm, compact, config.BLOCK_EXACT_LIMIT,
                  source, country_norm, name[:5], config.BLOCK_PREFIX_POOL_SIZE,
                  source, country_norm, address, config.BLOCK_EXACT_LIMIT)
        prefix_rows = []
        for row in con.execute(_EXACT_BLOCK_SQL, params):
            if row["rule"] == "name_prefix":
                prefix_rows.append(row)
                continue
            rid = int(row["rowid"])
            if rid not in candidates:
                candidates[rid] = {**dict(row), "rules": set()}
            candidates[rid]["rules"].add(row["rule"])
        if prefix_rows:
            ranked_prefix = process.extract(name, [row["name_norm"] for row in prefix_rows],
                                            scorer=fuzz.token_set_ratio, limit=config.BLOCK_PREFIX_LIMIT)
            for _, _, index in ranked_prefix:
                row = prefix_rows[index]
                rid = int(row["rowid"])
                if rid not in candidates:
                    candidates[rid] = {**dict(row), "rules": set()}
                candidates[rid]["rules"].add("name_prefix")
        _fts_field_candidates(con, source, "name_norm", name, country_norm,
                              config.BLOCK_FTS_NAME_TOP_K, "fts_name", candidates)
        _fts_field_candidates(con, source, "address_norm", address, country_norm,
                              config.BLOCK_FTS_ADDRESS_TOP_K, "fts_address", candidates)
    return list(candidates.values())
