import re

with open('paper/references.bib', 'r', encoding='utf-8') as f:
    text = f.read()

entries = []
raw_entries = re.split(r'(?m)^@', text)
for raw in raw_entries:
    if not raw.strip():
        continue
    m = re.match(r'(\w+)\s*\{\s*([^,]+),(.*)', raw, re.DOTALL)
    if not m:
        continue
    etype, key, body = m.group(1), m.group(2).strip(), m.group(3)
    
    fields = {}
    idx = 0
    while idx < len(body):
        f_match = re.search(r'([a-zA-Z_-]+)\s*=\s*\{', body[idx:])
        if not f_match:
            break
        fname = f_match.group(1).lower()
        start = idx + f_match.end()
        depth = 1
        pos = start
        while pos < len(body) and depth > 0:
            if body[pos] == '{':
                depth += 1
            elif body[pos] == '}':
                depth -= 1
            pos += 1
        val = body[start:pos-1].strip()
        fields[fname] = val
        idx = pos
    entries.append((key, etype, fields))

entries.sort(key=lambda x: x[0].lower())
print(f"Parsed {len(entries)} entries successfully!")

bbl_lines = [
    f"\\begin{{thebibliography}}{{{len(entries)}}}",
    "\\providecommand{\\natexlab}[1]{#1}",
    "\\providecommand{\\url}[1]{\\texttt{#1}}",
    ""
]

for key, etype, f in entries:
    author = f.get('author', 'Unknown')
    # clean author representation
    year = f.get('year', '2024')
    title = f.get('title', '')
    
    # Clean label
    authors_list = [a.strip() for a in author.split(' and ')]
    if len(authors_list) == 1:
        cite_label = f"{authors_list[0]}({year})"
        formatted_authors = authors_list[0]
    elif len(authors_list) == 2:
        cite_label = f"{authors_list[0]} and {authors_list[1]}({year})"
        formatted_authors = f"{authors_list[0]} and {authors_list[1]}"
    else:
        first = authors_list[0].replace('{', '').replace('}', '')
        cite_label = f"{first} et~al.({year})"
        formatted_authors = f"{authors_list[0]} et~al."
    
    # Clean braces for label
    cite_label = cite_label.replace('{', '').replace('}', '')
    
    bbl_lines.append(f"\\bibitem[{cite_label}]{{{key}}}")
    bbl_lines.append(f"{formatted_authors} {year}.")
    bbl_lines.append(f"\\newblock {title}.")
    
    venue = f.get('journal') or f.get('booktitle') or f.get('howpublished') or f.get('institution') or f.get('publisher') or ''
    vol = f.get('volume', '')
    num = f.get('number', '')
    pages = f.get('pages', '')
    
    extra = []
    if venue:
        extra.append(f"\\emph{{{venue}}}")
    if vol:
        if num:
            extra.append(f"{vol}({num})")
        else:
            extra.append(f"{vol}")
    if pages:
        extra.append(f"pp. {pages}")
    
    if extra:
        extra_joined = ", ".join(extra)
        bbl_lines.append(f"\\newblock {extra_joined}.")
    
    doi = f.get('doi', '')
    url = f.get('url', '')
    if doi:
        bbl_lines.append(f"\\newblock DOI: \\href{{https://doi.org/{doi}}}{{{doi}}}.")
    elif url:
        bbl_lines.append(f"\\newblock URL: \\url{{{url}}}.")
    
    bbl_lines.append("")

bbl_lines.append("\\end{thebibliography}")

full_bbl = "\n".join(bbl_lines)
with open('paper/main.bbl', 'w', encoding='utf-8') as out:
    out.write(full_bbl)

print("Wrote paper/main.bbl successfully!")
