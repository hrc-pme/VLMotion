import argparse
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from langchain_community.vectorstores import Chroma

def query_vector_db(db_path, query, k=3):
    print(f"正在載入 Chroma 資料庫: {db_path} ...")
    
    # 必須使用和建立時一模一樣的 Embedding 模型
    embeddings_model = HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese" # 如果您在轉換時有更改模型，這裡也要改
    )
    
    # 載入現有的向量庫
    vector_store = Chroma(
        persist_directory=db_path, 
        embedding_function=embeddings_model
    )
    
    print(f"\n正在搜尋問題: 「{query}」")
    print("-" * 50)
    
    # 進行相似度檢索 (尋找最相關的 k 個區塊)
    docs = vector_store.similarity_search(query, k=k)
    
    if not docs:
        print("找不到相關的內容。")
        return
        
    for i, doc in enumerate(docs):
        print(f"【相關內容片段 {i+1}】 (來源檔案: {doc.metadata.get('source', '未知')}, 頁數: {doc.metadata.get('page', '未知')})")
        print(doc.page_content)
        print("-" * 50)

def get_rag_context(db_path, query, k=3, threshold=300):
    """
    獲取 RAG 上下文，並加入信心度（距離）過濾。
    :param threshold: 距離門檻。數值越小要求越嚴格。Chroma 預設使用 L2 距離。
    """
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma

        embeddings_model = HuggingFaceEmbeddings(
            model_name="shibing624/text2vec-base-chinese"
        )
        
        vector_store = Chroma(
            persist_directory=db_path, 
            embedding_function=embeddings_model
        )
        
        # 1. 關鍵修改：改用 with_score 來獲取每筆資料的「距離分數」
        # 回傳格式會是 List[Tuple[Document, float]]
        docs_and_scores = vector_store.similarity_search_with_score(query, k=k)
        
        if not docs_and_scores:
            return ""
            
        valid_contexts = []
        
        print(f"\n=== RAG 檢索啟動 (問題: {query}) ===")
        # 2. 進行信心度篩選
        for doc, score in docs_and_scores:
            # 印出前 20 個字來觀察是哪一段 SOP，並顯示分數
            preview_text = doc.page_content[:20].replace('\n', ' ')
            print(f"[檢測] 分數: {score:.4f} | 預覽: {preview_text}...")
            
            # 分數 (距離) 小於門檻，代表「夠相似」，准許放行
            if score < threshold:
                valid_contexts.append(doc.page_content)
            else:
                print(f"  -> [阻擋] 距離過遠 (>{threshold})，判定為不相關。")
                
        print("===================================\n")
        
        # 3. 判斷篩選後是否還有剩下的資料
        if not valid_contexts:
            # 如果全部都被阻擋了，回傳空字串，讓 LLM 自己發揮
            return ""
            
        # 4. 將通過審核的資料組合起來
        context = "【參考資料 SOP】：\n"
        for content in valid_contexts:
            context += f"{content}\n\n"
            
        return context.strip()
        
    except Exception as e:
        print(f"RAG Error: {e}")
        return ""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="查詢已建立的 Chroma 向量資料庫")
    parser.add_argument("query", help="您想在文件中搜尋的問題或關鍵字")
    parser.add_argument("--db-path", default="./chroma_db", help="Chroma 資料庫存放的資料夾路徑 (預設: ./chroma_db)")
    parser.add_argument("-k", type=int, default=3, help="要返回最相關的片段數量 (預設: 3)")
    
    args = parser.parse_args()
    query_vector_db(args.db_path, args.query, args.k)
