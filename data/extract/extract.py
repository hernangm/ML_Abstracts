import os
import json
import re
from docx import Document

def parse_docx(input_file, output_file):
    # Load the document
    doc = Document(input_file)

    # Collect text paragraphs
    paragraphs = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:  # skip empty lines
            paragraphs.append(text)

    abstracts = []
    buffer = []
    for para in paragraphs:
        # Detect abstract start: a line with exactly 4 digits
        if re.match(r'^\d{4}$', para):
            if buffer:  # if we already collected one abstract, save it
                abstracts.append(buffer)
                buffer = []
        buffer.append(para)
    if buffer:
        abstracts.append(buffer)

    result = []
    for abs_paras in abstracts:
        # First element should be ID
        abs_id = abs_paras[0]
        # Second element should be the title
        title = abs_paras[1] if len(abs_paras) > 1 else ""
        # Body = everything after the first 4 header lines
        body = "\n".join(abs_paras[4:]).strip()
        result.append({
            "id": int(abs_id),
            "title": title,
            "body": body
        })

    # Write JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    print(f"Parsed {len(result)} abstracts. JSON saved to {output_file}")



if __name__ == "__main__":


    # __file__ is the current script's path
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # go one level up and into "data"
    doc_path = os.path.join(BASE_DIR, "raw_data", "Originales_Tomo1.docx")

    print(doc_path)
    parse_docx(doc_path, "abstracts2.json")