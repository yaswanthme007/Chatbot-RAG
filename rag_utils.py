import tempfile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from PyPDF2 import PdfReader  # ✅ for PDF text extraction


def process_uploaded_files(uploaded_files):
    docs = []
    for file in uploaded_files:
        # Save uploaded file to a temporary file
        suffix = ".pdf" if file.name.lower().endswith(".pdf") else ".txt"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        text = ""
        if suffix == ".pdf":
            # ✅ Extract text from PDF
            reader = PdfReader(tmp_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        else:
            # ✅ Read plain text file
            with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()

        if not text:
            print(f"⚠️ Skipped empty file: {file.name}")
            continue

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)

        if not chunks:
            print(f"⚠️ No chunks created for: {file.name}")
            continue

        docs.extend([Document(page_content=chunk, metadata={"source": file.name}) for chunk in chunks])

    if not docs:
        raise ValueError("❌ No valid text found in uploaded files. Please check the content.")

    # HuggingFace embeddings (no API key required)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create FAISS vectorstore
    return FAISS.from_documents(docs, embeddings)


def get_retriever(vectorstore):
    return vectorstore.as_retriever(search_kwargs={"k": 3})
