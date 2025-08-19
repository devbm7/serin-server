import PyPDF2

pdf_file = open("Resume v3.01.pdf", "rb")
pdf_reader = PyPDF2.PdfReader(pdf_file)

content = ""

for page in pdf_reader.pages:
    content += page.extract_text()

print(content)
pdf_file.close()
