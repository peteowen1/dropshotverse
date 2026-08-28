"""
Fetch Aneeshers/tennis-sackmann-archive from both its GitHub repo and its
HuggingFace dataset mirror, verifying byte-for-byte agreement.

Ticket: 13-snapshot-base-archive.md
No git operations are performed here (plain HTTP downloads only) -- this is
a local-filesystem freeze, not a new repo.
"""
import json
import os
import sys
import time
import hashlib
import urllib.request
import urllib.error

REPO = "Aneeshers/tennis-sackmann-archive"
GH_TREE_JSON = r"C:\Users\peteo\AppData\Local\Temp\claude\C--dev-dropshotverse\da4071a3-f79b-4a5c-89ac-53452c974648\scratchpad\gh_tree.json"
HF_TREE_JSON = r"C:\Users\peteo\AppData\Local\Temp\claude\C--dev-dropshotverse\da4071a3-f79b-4a5c-89ac-53452c974648\scratchpad\hf_tree.json"
GH_COMMIT = "83733587353df8a41f2fd4f516147d5aa83f5a8d"
HF_COMMIT_META = "8ac86f7c62889b5d5c4cfec4c6403edcb8c5d993"

OUT_ROOT = r"C:\dev\dropshotverse\dropshotdata"
GH_OUT = os.path.join(OUT_ROOT, "raw", "github")
HF_OUT = os.path.join(OUT_ROOT, "raw", "huggingface")
MANIFEST_OUT = os.path.join(OUT_ROOT, "manifest", "manifest.json")

GH_RAW_BASE = f"https://raw.githubusercontent.com/{REPO}/{GH_COMMIT}/"
HF_RAW_BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main/"


def load_gh_files():
    d = json.load(open(GH_TREE_JSON, encoding="utf-8"))
    return {t["path"]: {"size": t.get("size", 0), "blob_sha": t["sha"]}
            for t in d["tree"] if t["type"] == "blob"}


def load_hf_files():
    d = json.load(open(HF_TREE_JSON, encoding="utf-8"))
    return {t["path"]: {"size": t.get("size", 0), "oid": t.get("oid"), "lfs": bool(t.get("lfs"))}
            for t in d if t["type"] == "file"}


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, dest, retries=4):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        return "cached"
    req = urllib.request.Request(url, headers={"User-Agent": "dropshotverse-archive-freeze/1.0"})
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as out:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
            return "downloaded"
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
            if os.path.exists(dest):
                os.remove(dest)
    raise RuntimeError(f"failed to download {url}: {last_err}")


def is_priority(path):
    """ATP+WTA main-tour match-level CSVs -- the priority layer per ticket 13."""
    fname = os.path.basename(path)
    if not (path.startswith("atp/") or path.startswith("wta/")):
        return False
    if "matches" not in fname:
        return False
    for excl in ("qual", "chall", "futures", "doubles"):
        if excl in fname:
            return False
    return True


def main():
    gh_files = load_gh_files()
    hf_files = load_hf_files()

    gh_paths = set(gh_files)
    hf_paths = set(hf_files)
    print(f"GitHub file count: {len(gh_paths)}")
    print(f"HuggingFace file count: {len(hf_paths)}")
    only_gh = gh_paths - hf_paths
    only_hf = hf_paths - gh_paths
    if only_gh:
        print("Files ONLY on GitHub:", sorted(only_gh))
    if only_hf:
        print("Files ONLY on HuggingFace:", sorted(only_hf))

    all_paths = sorted(gh_paths | hf_paths)
    priority = [p for p in all_paths if is_priority(p)]
    rest = [p for p in all_paths if not is_priority(p)]

    priority_size = sum(gh_files.get(p, {}).get("size", 0) for p in priority)
    rest_size = sum(gh_files.get(p, {}).get("size", 0) for p in rest)
    print(f"Priority layer (ATP+WTA main-tour match CSVs): {len(priority)} files, "
          f"{priority_size/1024/1024:.1f} MB (GitHub-side size)")
    print(f"Remaining layer: {len(rest)} files, {rest_size/1024/1024:.1f} MB (GitHub-side size)")

    manifest = []
    ordered = priority + rest
    for i, path in enumerate(ordered, 1):
        entry = {"path": path}
        gh_dest = os.path.join(GH_OUT, path)
        hf_dest = os.path.join(HF_OUT, path)

        gh_status = hf_status = None
        gh_err = hf_err = None

        if path in gh_files:
            try:
                gh_status = download(GH_RAW_BASE + path, gh_dest)
            except Exception as e:
                gh_err = str(e)
        if path in hf_files:
            try:
                hf_status = download(HF_RAW_BASE + path, hf_dest)
            except Exception as e:
                hf_err = str(e)

        gh_sha256 = sha256_of(gh_dest) if os.path.exists(gh_dest) else None
        hf_sha256 = sha256_of(hf_dest) if os.path.exists(hf_dest) else None
        gh_size_actual = os.path.getsize(gh_dest) if os.path.exists(gh_dest) else None
        hf_size_actual = os.path.getsize(hf_dest) if os.path.exists(hf_dest) else None

        entry.update({
            "priority": path in priority,
            "gh_status": gh_status, "gh_error": gh_err,
            "hf_status": hf_status, "hf_error": hf_err,
            "gh_size_bytes": gh_size_actual, "hf_size_bytes": hf_size_actual,
            "gh_sha256": gh_sha256, "hf_sha256": hf_sha256,
            "mirrors_agree": (gh_sha256 is not None and gh_sha256 == hf_sha256),
            "fetched_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        manifest.append(entry)
        if i % 25 == 0 or i == len(ordered):
            print(f"  [{i}/{len(ordered)}] {path}")

    os.makedirs(os.path.dirname(MANIFEST_OUT), exist_ok=True)
    with open(MANIFEST_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "source_repo_github": f"https://github.com/{REPO}",
            "source_repo_huggingface": f"https://huggingface.co/datasets/{REPO}",
            "github_commit_sha": GH_COMMIT,
            "huggingface_commit_sha": HF_COMMIT_META,
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "files": manifest,
        }, f, indent=2)
    print(f"\nManifest written: {MANIFEST_OUT}")

    mismatches = [m for m in manifest if m["gh_sha256"] and m["hf_sha256"] and m["gh_sha256"] != m["hf_sha256"]]
    missing_one_side = [m for m in manifest if not (m["gh_sha256"] and m["hf_sha256"])]
    print(f"Mirror checksum mismatches: {len(mismatches)}")
    for m in mismatches:
        print("  MISMATCH:", m["path"], m["gh_sha256"], m["hf_sha256"])
    print(f"Files present on only one mirror: {len(missing_one_side)}")
    for m in missing_one_side:
        print("  ONE-SIDED:", m["path"], "gh:", bool(m["gh_sha256"]), "hf:", bool(m["hf_sha256"]))


if __name__ == "__main__":
    main()
