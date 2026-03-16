# ocr.py
# Usage:
#   python ocr.py gemma3 data/test_images
#   python ocr.py gemma3 data/test_images --recursive
#   python ocr.py gemma3 data/test_images --prompt "Extract all readable text. Return plain text only."

import sys
import re
from pathlib import Path
from collections import defaultdict
from ollama import Client

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULT_PROMPT = "Extract all readable text from this image. Return plain text only."
DEFAULT_TIMEOUT = 120.0

# Finds occurrences like "_page12" or "-page12" anywhere in the stem
PAGE_PAT = re.compile(r"(?i)([_-])page(?P<page>\d+)\b")


def ocr_image(model: str, image_path: Path, prompt: str, timeout: float = DEFAULT_TIMEOUT) -> str:
    client = Client(timeout=timeout)
    resp = client.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [str(image_path)],
            }
        ],
        options={"temperature": 0.0},
    )

    if hasattr(resp, "message") and hasattr(resp.message, "content"):
        return resp.message.content or ""
    if isinstance(resp, dict):
        return (resp.get("message") or {}).get("content", "") or ""
    return str(resp)


def doc_key_and_page(stem: str) -> tuple[str, int]:
    """
    Robustly returns (doc_key, page_num).

    Works for:
      Doc_page1
      Doc_page01
      Doc_page1_rotated
      Doc-page2-anything
      Doc_page3 (1)
    by taking the LAST occurrence of *_page<digits>* in the filename stem.
    """
    last = None
    for m in PAGE_PAT.finditer(stem):
        last = m

    if not last:
        return stem, 0

    page_num = int(last.group("page"))
    doc = stem[: last.start()]  # everything before "_pageN"
    doc = re.sub(r"[_-]+$", "", doc).strip()  # trim trailing separators
    if not doc:
        doc = stem  # fallback safety
    return doc, page_num


def main():
    if len(sys.argv) < 3:
        print("Usage: python ocr.py <model> <images_folder> [--recursive] [--prompt <text>]")
        sys.exit(1)

    model = sys.argv[1]
    in_dir = Path(sys.argv[2]).resolve()
    recursive = "--recursive" in sys.argv

    prompt = DEFAULT_PROMPT
    if "--prompt" in sys.argv:
        i = sys.argv.index("--prompt")
        if i + 1 < len(sys.argv):
            prompt = sys.argv[i + 1]

    if not in_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {in_dir}")

    out_dir = Path(str(in_dir) + "_ocrtext")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect images
    if recursive:
        images = [p for p in in_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    else:
        images = [p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]

    if not images:
        print(f"No images found in {in_dir} (extensions: {sorted(IMAGE_EXTS)})")
        sys.exit(0)

    # Group images by doc key and page number
    groups = defaultdict(list)
    for img in images:
        doc_key, page_num = doc_key_and_page(img.stem)
        groups[doc_key].append((page_num, img))

    doc_names = sorted(groups.keys())

    ok_docs = 0
    fail_pages = 0
    total_pages = 0

    for doc in doc_names:
        pages = sorted(groups[doc], key=lambda x: (x[0], x[1].name))
        merged_parts = []

        for page_num, img in pages:
            total_pages += 1
            try:
                text = ocr_image(model=model, image_path=img, prompt=prompt).strip()

                # If we found a page number, label as PAGE N; otherwise just label IMAGE
                if page_num > 0:
                    merged_parts.append(f"\n\n===== PAGE {page_num} ({img.name}) =====\n{text}\n")
                else:
                    merged_parts.append(f"\n\n===== IMAGE ({img.name}) =====\n{text}\n")

                print(f"[OK] {doc} :: {img.name}")
            except Exception as e:
                fail_pages += 1
                merged_parts.append(
                    f"\n\n===== PAGE {page_num if page_num>0 else '?'} ({img.name}) OCR FAILED =====\n{e}\n"
                )
                print(f"[FAIL] {doc} :: {img} -> {e}")

        out_path = out_dir / f"{doc}.txt"
        out_path.write_text("".join(merged_parts).lstrip(), encoding="utf-8")
        ok_docs += 1
        print(f"[MERGED] {doc} -> {out_path.name} ({len(pages)} page(s))")

    print("\nDone.")
    print(f"Documents merged: {ok_docs}")
    print(f"Pages processed: {total_pages}")
    print(f"Page failures: {fail_pages}")
    print(f"Output folder: {out_dir}")


if __name__ == "__main__":
    main()
