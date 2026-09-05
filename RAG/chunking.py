from pypdf import PdfReader

# Load the PDF file
reader = PdfReader("D:\\Learning\\artificial-intelligent\\leave_policy.pdf")

# Get total number of pages
total_pages = len(reader.pages)
# print(f"Total Pages: {total_pages}")

# Extract and print text from the first page (index 0)
first_page = reader.pages[0:8]
# print("1st page:", first_page)
for page in first_page:
    # print("page:", page)
    text = page.extract_text()
    split_text = text.split("\n")
    # print(split_text)
    chunks = []
    for line in range(0, len(split_text), 2):
        chunk = " ".join(split_text[line : line + 2])
        print("chunk:", chunk)
        chunks.append(chunk)
    for chunk in chunks:
        print(f"Chunk : {chunk}")
