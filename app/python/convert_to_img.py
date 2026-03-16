import os
import re
from pathlib import Path
from typing import Optional, Sequence
import fitz  # PyMuPDF


def sanitize_stem(stem: str) -> str:
    stem = (stem or "").strip()
    stem = re.sub(r"\s+", "_", stem)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem or "document"


def build_unique_base(pdf_file: Path, root: Path) -> str:
    """
    Create a unique base name using relative path:
      root/subA/contract.pdf -> subA__contract
    """
    rel = pdf_file.relative_to(root)
    parts = [sanitize_stem(p) for p in rel.with_suffix("").parts]  # drop .pdf
    return "__".join(parts)


def convert_pdf2img(
    input_file: str,
    out_dir: str,
    base_override: str | None = None,
    pages: Optional[Sequence[int]] = None,
    zoom: float = 2.0,
    rotate: int = 0,
    pad: int = 4,
) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)

    pdf = fitz.open(input_file)
    output_files: list[str] = []
    try:
        total_pages = pdf.page_count
        width = max(pad, len(str(total_pages)))

        base = sanitize_stem(base_override) if base_override else sanitize_stem(Path(input_file).stem)

        for pg in range(total_pages):
            if pages is not None and (pg + 1) not in pages:
                continue

            page = pdf[pg]
            mat = fitz.Matrix(zoom, zoom).prerotate(rotate)
            pix = page.get_pixmap(matrix=mat, alpha=False)  # type: ignore

            page_str = str(pg + 1).zfill(width)
            out_path = Path(out_dir) / f"{base}_page{page_str}.png"
            pix.save(str(out_path))
            output_files.append(str(out_path))
    finally:
        pdf.close()

    return output_files


def batch_convert_folder(input_dir: str, recursive: bool = False, zoom: float = 2.0) -> dict[str, list[str]]:
    input_path = Path(input_dir).resolve()
    if not input_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {input_path}")

    out_dir = str(Path(str(input_path) + "_images"))
    os.makedirs(out_dir, exist_ok=True)

    pdf_paths = sorted(input_path.rglob("*.pdf")) if recursive else sorted(p for p in input_path.glob("*.pdf") if p.is_file())

    results: dict[str, list[str]] = {}
    for pdf_file in pdf_paths:
        try:
            base = build_unique_base(pdf_file, input_path) if recursive else sanitize_stem(pdf_file.stem)
            outputs = convert_pdf2img(str(pdf_file), out_dir=out_dir, base_override=base, zoom=zoom)
            results[str(pdf_file)] = outputs
            print(f"[OK] {pdf_file} -> {len(outputs)} images")
        except Exception as e:
            print(f"[FAIL] {pdf_file} -> {e}")

    print(f"\nDone. Output folder: {out_dir}")
    return results


if __name__ == "__main__":
    import sys
    input_dir = sys.argv[1]
    recursive = "--recursive" in sys.argv
    batch_convert_folder(input_dir, recursive=recursive, zoom=2.0)
