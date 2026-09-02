import pdfplumber, sys
sys.stdout.reconfigure(encoding='utf-8')
with pdfplumber.open("ThieuDangHung_22050015 - DeCuongDATN.pdf") as pdf:
    for i, page in enumerate(pdf.pages):
        print(f"--- Page {i+1} ---")
        t = page.extract_text()
        if t:
            print(t)
