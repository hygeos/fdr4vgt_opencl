#!/usr/bin/env python3
"""Post-process DOCX for equation alignment and caption styling."""

from __future__ import annotations

import shutil
import tempfile
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import xml.etree.ElementTree as ET
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

ET.register_namespace("w", W_NS)
ET.register_namespace("m", M_NS)


def _set_caption_style(xml: str) -> tuple[str, int]:
    ns = {"w": W_NS}
    root = ET.fromstring(xml)
    changed = 0

    # Promote explicit caption paragraphs ("Figure N:" / "Table N:")
    # to Word Caption style so LOF/LOT fields can populate.
    for p in root.findall(".//w:p", ns):
        text = "".join((t.text or "") for t in p.findall(".//w:t", ns)).strip()
        if not re.match(r"^(Figure|Table)\s+\d+\s*:", text):
            continue

        p_pr = p.find("w:pPr", ns)
        if p_pr is None:
            p_pr = ET.Element(f"{{{W_NS}}}pPr")
            p.insert(0, p_pr)

        p_style = p_pr.find("w:pStyle", ns)
        if p_style is None:
            p_style = ET.SubElement(p_pr, f"{{{W_NS}}}pStyle")

        val_attr = f"{{{W_NS}}}val"
        if p_style.get(val_attr) != "Caption":
            p_style.set(val_attr, "Caption")
            changed += 1

    return ET.tostring(root, encoding="unicode"), changed


def patch_document_xml(xml: str) -> tuple[str, int, int]:
    target = '<m:oMathParaPr><m:jc m:val="center" /></m:oMathParaPr>'
    replacement = '<m:oMathParaPr><m:jc m:val="right" /></m:oMathParaPr>'
    count = xml.count(target)
    if count:
        xml = xml.replace(target, replacement)
    xml, caption_count = _set_caption_style(xml)
    return xml, count, caption_count


def fix_docx(docx_path: Path) -> tuple[int, int]:
    with tempfile.TemporaryDirectory(prefix="docx_fix_") as tmpdir:
        tmp = Path(tmpdir)
        with ZipFile(docx_path, "r") as zin:
            zin.extractall(tmp)

        doc_xml_path = tmp / "word" / "document.xml"
        xml = doc_xml_path.read_text(encoding="utf-8")
        patched, eq_count, caption_count = patch_document_xml(xml)
        doc_xml_path.write_text(patched, encoding="utf-8")

        out_path = docx_path.with_suffix(docx_path.suffix + ".tmp")
        with ZipFile(out_path, "w", ZIP_DEFLATED) as zout:
            for p in sorted(tmp.rglob("*")):
                if p.is_file():
                    zout.write(p, p.relative_to(tmp).as_posix())

        shutil.move(out_path, docx_path)
        return eq_count, caption_count


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Right-align OMML equation paragraphs in a DOCX file.")
    parser.add_argument("docx", type=Path, help="Path to DOCX file")
    args = parser.parse_args()

    eq_changed, caption_changed = fix_docx(args.docx)
    print(f"patched_equations={eq_changed}")
    print(f"patched_captions={caption_changed}")


if __name__ == "__main__":
    main()
