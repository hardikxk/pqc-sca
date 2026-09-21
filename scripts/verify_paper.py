"""
Verify research paper integrity:
1. Balanced LaTeX curly braces { }
2. All \\cite{...} keys exist as \\bibitem{...} in paper/main.tex or paper/references.bib
"""
import os
import re
import sys

def verify_paper(tex_path="paper/main.tex"):
    if not os.path.exists(tex_path):
        print(f"Error: {tex_path} not found.")
        sys.exit(1)

    with open(tex_path, "r", encoding="utf-8") as f:
        tex = f.read()

    # 1. Balanced braces check
    open_braces = tex.count("{")
    close_braces = tex.count("}")
    if open_braces != close_braces:
        print(f"Error: Unbalanced braces in {tex_path}: {open_braces} open vs {close_braces} close.")
        sys.exit(1)

    # 2. Citations check
    # Match \cite{...}, \citep{...}, \citet{...}
    cites = set(re.findall(r"\\cite[pt]?\{([^}]+)\}", tex))
    all_cites = set()
    for c in cites:
        for k in c.split(","):
            k = k.strip()
            if k:
                all_cites.add(k)

    # Match \bibitem{...} or \bibitem[...]...
    bibitems = set(re.findall(r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}", tex))
    
    # Also check references.bib if bibitems are in bib file
    bib_path = "paper/references.bib"
    if os.path.exists(bib_path):
        with open(bib_path, "r", encoding="utf-8") as f:
            bib_content = f.read()
        bib_entries = set(re.findall(r"@\w+\s*\{\s*([^,]+),", bib_content))
        all_available_refs = bibitems | bib_entries
    else:
        all_available_refs = bibitems

    missing = all_cites - all_available_refs
    if missing:
        print(f"Error: Missing references in {tex_path}: {sorted(missing)}")
        sys.exit(1)

    print(f"[OK] Research paper verified successfully: {len(all_cites)} citations checked, 0 missing, braces balanced ({open_braces}).")

if __name__ == "__main__":
    verify_paper()
