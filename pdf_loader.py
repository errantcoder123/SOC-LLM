from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader

pdf_path = r"D:\Langchain\Langchain_models\document_loader\paper.pdf"

structured_loader = PyPDFLoader(pdf_path)
raw_loader = UnstructuredPDFLoader(pdf_path)

document_chunks = raw_loader.load()

print(document_chunks[0].page_content)
