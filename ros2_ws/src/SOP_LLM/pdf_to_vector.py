import argparse
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def pdf_to_vector_db(pdf_path, db_path):
    if not os.path.exists(pdf_path):
        print(f"錯誤: 找不到 PDF 檔案 '{pdf_path}'")
        return

    print(f"[1/4] 正在載入 PDF 檔案: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"      成功載入 {len(documents)} 頁文件。")

    print("[2/4] 正在將文字切分為區塊 (Chunks)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    print(f"      成功切分為 {len(chunks)} 個文字區塊。")

    print("[3/4] 正在初始化 Embedding 模型...")
    # 若需處理中文，可換成這類的多語言模型: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    # 這裡預設使用輕量的中英可用模型: shibing624/text2vec-base-chinese
    embeddings_model = HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese"
    )

    print(f"[4/4] 正在建立 Chroma 向量資料庫並儲存至 '{db_path}'...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=db_path
    )
    print("完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="將 PDF 轉換為 Chroma 向量資料庫")
    parser.add_argument("pdf_path", help="要轉換的 PDF 檔案路徑")
    parser.add_argument("--db-path", default="./chroma_db", help="Chroma 資料庫存放的資料夾路徑 (預設: ./chroma_db)")
    
    args = parser.parse_args()
    pdf_to_vector_db(args.pdf_path, args.db_path)
