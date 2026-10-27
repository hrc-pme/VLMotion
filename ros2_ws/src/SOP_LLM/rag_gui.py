import sys
import os
import threading
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QPushButton, QTextEdit
)
from PyQt5.QtCore import pyqtSignal, QObject

# 匯入 LangChain 相關套件
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_core.prompts import PromptTemplate
import torch

class RAGWorker(QObject):
    """
    因為 LLM 推論會卡住主程式，我們用另外的執行緒和 PyQt Signal 來處理
    """
    finished = pyqtSignal(str)
    status_update = pyqtSignal(str)

    def __init__(self, db_path):
        super().__init__()
        self.db_path = db_path
        self.vector_store = None
        self.llm = None
        self.query_text = ""

    def process_query(self):
        lower_query = self.query_text.lower()
        
        # 1. 整理關鍵字清單
        # 代表要「執行清點任務」的強烈指令
        action_keywords = ["開始清點", "盤點桌上", "count the items", "start counting", "count it"]
        
        # 代表只是在「詢問資訊」的排除字眼
        inquiry_keywords = ["sequence", "order", "how to", "順序", "怎麼", "sop", "規定"]

        # 2. 核心判斷邏輯
        # 條件 A：句子裡包含詢問字眼 -> 絕對是問答模式，跳過 JSON 觸發
        is_inquiry = any(word in lower_query for word in inquiry_keywords)
        
        # 條件 B：句子裡包含明確的執行指令，或者是單純只有 "count" 且沒有詢問字眼
        is_task = False
        if not is_inquiry:
            if any(word in lower_query for word in action_keywords):
                is_task = True
            elif "count" in lower_query and len(lower_query.split()) <= 3:
                # 如果只輸入 "count" 或 "count please" 這種極短句，也當作任務
                is_task = True
            elif "清點" in self.query_text and ("json" in lower_query or "schema" in lower_query or "格式" in self.query_text):
                # 保留您原本的強制 JSON 觸發邏輯
                is_task = True

        # 3. 執行分流
        if is_task:
            try:
                self.status_update.emit("🎯 偵測到清點指令：啟動任務模式")
                import json
                file_path = os.path.join(os.path.dirname(__file__), "json_sop", "counting.json")
                with open(file_path, 'r', encoding='utf-8') as f:
                    sop_data = json.load(f)
                
                # 這裡直接輸出 JSON 結構，或者您未來可以把這裡改成呼叫 VLM 進行影像辨識
                response_json = json.dumps(sop_data.get("output_schema_example", {}), indent=2, ensure_ascii=False)
                
                self.finished.emit(f"```json\n{response_json}\n```")
                self.status_update.emit("✅ 清點任務完成！")
                return
            except Exception as e:
                self.finished.emit(f"生成 JSON 回答時發生錯誤: {str(e)}")
                return

        # ==========================================
        # 如果不是任務模式 (is_task == False)，就繼續往下執行您原本的 RAG 諮詢邏輯
        # ==========================================
        
        if not self.load_models():
            return

        self.status_update.emit(f"正在知識庫中搜尋相關 SOP 段落...")
        # 增加 k 值確保能撈到足夠的上下文 (第一年到第四年的跨章節資訊)
        docs = self.vector_store.similarity_search(self.query_text, k=6)
        
        if not docs:
            self.finished.emit("無法在資料庫中找到相關的內容。")
            return
            
        context = "\n\n".join([f"片段 {i+1}: {doc.page_content}" for i, doc in enumerate(docs)])
        
        self.status_update.emit("正在思考並生成口語化回答，請稍候...")
        
        # 【Prompt 升級】：賦予護理助理人設，並制定嚴格的行為邊界
        template = """【你的角色】
你是一個在手術室協助護理人員的「智慧型機器人助理」。
⚠️ 目前為純文字測試階段，完全看不到畫面。

【你的說話風格】
1. 專業、精準、直接回答問題。
2. 【嚴禁廢話與過度解釋】：護理人員問什麼，你就只回答該情境的規定。絕對不可以把其他無關的規定念出來！講完就立刻結束對話。

【你的大腦邏輯（檢索到的 SOP 規範）】
{context}

【情境對答指南】（請嚴格判斷護理師的問題屬於哪一種，並『只回答』該情境的內容）
- 情境 A【被問到清點順序、被要求跳著算、或詢問 counting sequence】：
  你必須說明順序規定，並完整回答：「根據醫療安全約束，我必須遵循『由上而下、由左而右或由小到大』的固定順序來清點，絕對不可隨意跳躍。」
- 情境 B【被問到器械重疊、看不清楚】：
  你必須回答：「我絕對不瞎猜數量。我會先自動控制機械手臂進行輕柔撥散與展開；若多次嘗試仍解不開，我才會用語音明確指示空間方位，請您協助。」
- 情境 C【被問到手伸過去、人員靠近】：
  你必須回答：「我會動態閃避，距離太近會進入安全暫停。等您離開且同時滿足『大於安全距離』與『維持一段時間』兩個條件後，我才會自主恢復作業。」
{query}

【你的口語化精確回答】：
"""
        prompt = PromptTemplate(template=template, input_variables=["context", "query"])
        final_prompt = prompt.format(context=context, query=self.query_text)
        
        try:
            messages = [
                {"role": "user", "content": final_prompt}
            ]
            
            tokenized_chat = self.llm.pipeline.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            response = self.llm.invoke(tokenized_chat)
            final_answer = response.split("assistant\n")[-1].strip()
            
            # 【雙重保護】：暴力替換掉殘存的數學語法，確保畫面上絕對不會顯示亂碼
            final_answer = final_answer.replace("$", "")
            final_answer = final_answer.replace("\\|", "|")
            final_answer = final_answer.replace("\\ge", " 大於或等於 ")
            final_answer = final_answer.replace("\\le", " 小於或等於 ")
            final_answer = final_answer.replace("p_{ee}", "機器手臂末端位置")
            final_answer = final_answer.replace("p_h", "人員位置")
            final_answer = final_answer.replace("d_{safe}", "安全門檻")
            
            self.finished.emit(final_answer)
            self.status_update.emit("✅ 回答完成！")
            
        except Exception as e:
            self.finished.emit(f"生成回答時發生錯誤: {str(e)}")


class RAGMainWindow(QMainWindow):
    def __init__(self, db_path):
        super().__init__()
        self.db_path = db_path
        self.initUI()
        
        # 建立背景執行緒處理 RAG
        self.rag_worker = RAGWorker(self.db_path)
        self.rag_worker.finished.connect(self.on_query_finished)
        self.rag_worker.status_update.connect(self.update_status)
        
        # 初始化模型 (在背景執行以免卡住 UI)
        threading.Thread(target=self.init_models, daemon=True).start()

    def initUI(self):
        self.setWindowTitle('智慧醫療機器人助理 (VLM-RAG 測試介面)')
        self.setMinimumSize(850, 650)

        # 主要 Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 狀態標籤
        self.status_label = QLabel("正在啟動系統...")
        self.status_label.setStyleSheet("color: #D32F2F; font-weight: bold; font-size: 14px;")
        main_layout.addWidget(self.status_label)

        # 顯示對話/結果的區域
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("font-size: 15px; padding: 10px; background-color: #F5F5F5; border-radius: 5px;")
        main_layout.addWidget(self.chat_display)

        # 輸入區域
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("請輸入護理人員的實務提問")
        self.input_field.setStyleSheet("font-size: 15px; padding: 8px; border: 1px solid #CCC; border-radius: 4px;")
        self.input_field.returnPressed.connect(self.submit_query)  # 按 Enter 也可以送出
        
        self.submit_btn = QPushButton('送出提問')
        self.submit_btn.setStyleSheet("font-size: 15px; padding: 8px 20px; background-color: #1976D2; color: white; border-radius: 4px; font-weight: bold;")
        self.submit_btn.clicked.connect(self.submit_query)
        self.submit_btn.setEnabled(False) # 模型載入前先禁用
        
        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.submit_btn)
        
        main_layout.addLayout(input_layout)

    def init_models(self):
        # 第一次載入模型
        success = self.rag_worker.load_models()
        if success:
            self.submit_btn.setEnabled(True)
            # 系統啟動完成時改變狀態列顏色
            self.status_label.setStyleSheet("color: #388E3C; font-weight: bold; font-size: 14px;")

    def update_status(self, text):
        self.status_label.setText(text)

    def submit_query(self):
        query = self.input_field.text().strip()
        if not query:
            return
            
        self.input_field.clear()
        self.submit_btn.setEnabled(False)
        self.chat_display.append(f"<div style='color: #1565C0;'>👩‍⚕️ <b>護理人員:</b> {query}</div><br>")
        
        # 將問題交給背景執行緒處理
        self.rag_worker.query_text = query
        threading.Thread(target=self.rag_worker.process_query, daemon=True).start()

    def on_query_finished(self, result):
        self.chat_display.append(f"<div style='color: #2E7D32;'>🤖 <b>機器人助理:</b><br>{result}</div><br>")
        self.chat_display.append("<hr>")
        self.submit_btn.setEnabled(True)

if __name__ == '__main__':
    # 這裡確保 Qt 能正確運作在 Linux 環境中
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    
    app = QApplication(sys.argv)
    
    # 強制設定字型，確保能正確顯示中文 (改用 TC 即 Traditional Chinese)
    font = app.font()
    font.setFamily("Noto Sans CJK TC")
    app.setFont(font)

    # 請確認您的 ChromaDB 路徑正確
    db_path = "/workspace/pdf_to_vector/chroma_db"
    
    window = RAGMainWindow(db_path)
    window.show()
    sys.exit(app.exec_())