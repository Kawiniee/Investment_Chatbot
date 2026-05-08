"""
AI Investment Advisor Chatbot
=============================
- Library: LangChain + LangGraph
- Embedding Model: Google Gemini (gemini-embedding-001)
- Vector Search: ChromaDB
- Tools: calculate_investment_return, search_investment_info, recommend_portfolio
- Logging: บันทึกในไฟล์ logs/agent_YYYYMMDD.log
- Web Interface: Gradio (localhost:7860)
"""

import os
import math
import logging
from datetime import datetime
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
import gradio as gr
import uuid

# =============================================================================
# 1. SETUP ENVIRONMENT & LOGGING
# =============================================================================
load_dotenv()

# Set API Key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")

# Create logs directory
os.makedirs("logs", exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/agent_{datetime.now().strftime("%Y%m%d")}.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info("=" * 60)
logger.info("AI Investment Advisor starting...")
logger.info("=" * 60)

# =============================================================================
# 2. INITIALIZE MODELS
# =============================================================================
try:
    # Initialize Chat Model (LLM)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    logger.info("Chat model initialized: gemini-2.5-flash")

    # Initialize Embedding Model
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )
    logger.info("Embedding model initialized: gemini-embedding-001")
except Exception as e:
    logger.error(f"Failed to initialize models: {e}")
    raise

# =============================================================================
# 3. SETUP VECTOR STORE (ChromaDB)
# =============================================================================
# Investment knowledge base
investment_docs = [
    Document(
        page_content="การลงทุนในหุ้น (Stock Investment) คือการซื้อหุ้นของบริษัทจดทะเบียน เพื่อเป็นเจ้าของบริษัทและรับผลตอบแทนจากการเติบโตของบริษัท มีความเสี่ยงสูงแต่ผลตอบแทนในระยะยาวดี",
        metadata={"type": "stocks", "category": "equity"}
    ),
    Document(
        page_content="กองทุนรวม (Mutual Fund) คือการรวมเงินลงทุนของนักลงทุนหลายคน ให้ผู้จัดการกองทุนบริหารจัดการ มีความเสี่ยงต่ำกว่าการลงทุนในหุ้นโดยตรง",
        metadata={"type": "funds", "category": "managed"}
    ),
    Document(
        page_content="พันธบัตรรัฐบาล (Government Bond) คือการลงทุนในหนี้สินของรัฐบาล มีความเสี่ยงต่ำ ผลตอบแทนคงที่ เหมาะสำหรับนักลงทุนที่ต้องการความมั่นคง",
        metadata={"type": "bonds", "category": "fixed_income"}
    ),
    Document(
        page_content="การลงทุนในทองคำ (Gold Investment) เป็นสินทรัพย์ปลอดภัยในยามวิกฤต มักเคลื่อนไหวตรงกันข้ามกับตลาดหุ้น ควรถือเป็นส่วนหนึ่งของพอร์ตการลงทุน",
        metadata={"type": "gold", "category": "commodity"}
    ),
    Document(
        page_content="การจัดพอร์ตการลงทุน (Asset Allocation) ควรกระจายความเสี่ยง เช่น 60% หุ้น 40% ตราสารหนี้ ขึ้นอยู่กับความเสี่ยงที่รับได้",
        metadata={"type": "strategy", "category": "portfolio"}
    ),
    Document(
        page_content="คำแนะนำสำหรับมือใหม่: เริ่มต้นลงทุนในกองทุนรวมดัชนี (Index Fund) ที่มีค่าธรรมเนียมต่ำ กระจายความเสี่ยงด้วยการซื้อสม่ำเสมอ (Dollar Cost Averaging)",
        metadata={"type": "advice", "category": "beginner"}
    ),
    Document(
        page_content="หลักการลงทุน: ลงทุนในสิ่งที่เข้าใจ กระจายความเสี่ยง ถือระยะยาว ไม่ลงทุนด้วยเงินที่ต้องการใช้เร็ว",
        metadata={"type": "principle", "category": "strategy"}
    ),
    Document(
        page_content="การลงทุนในบริษัทที่มีปันผลสูง (Dividend Stocks) เหมาะสำหรับผู้ต้องการรายได้ประจำ มีความเสี่ยงปานกลาง",
        metadata={"type": "dividend", "category": "equity"}
    )
]

# Create ChromaDB vector store
vectorstore = Chroma.from_documents(
    documents=investment_docs,
    embedding=embeddings,
    collection_name="investment_knowledge"
)
logger.info(f"ChromaDB initialized with {len(investment_docs)} documents")

# =============================================================================
# 4. CREATE TOOLS
# =============================================================================
@tool
def calculate_investment_return(
    principal: float,
    rate: float,
    years: float
) -> float:
    """คำนวณผลตอบแทนจากการลงทุนแบบทบต้น (Compound Interest)

    Args:
        principal: เงินลงทุนเริ่มต้น (บาท)
        rate: อัตราผลตอบแทนต่อปี (%)
        years: จำนวนปีที่ลงทุน

    Returns:
        จำนวนเงินรวมที่ได้รับ
    """
    result = principal * math.pow(1 + rate/100, years)
    logger.info(f"Calculated: principal={principal}, rate={rate}%, years={years}, result={result}")
    return round(result, 2)

@tool
def search_investment_info(query: str) -> str:
    """ค้นหาข้อมูลเกี่ยวกับการลงทุนจากฐานความรู้

    Args:
        query: คำถามเกี่ยวกับการลงทุน

    Returns:
        ข้อมูลที่เกี่ยวข้อง
    """
    docs = vectorstore.similarity_search(query, k=3)
    result = "\n\n".join([d.page_content for d in docs])
    logger.info(f"Search: '{query}' found {len(docs)} documents")
    return result

@tool
def recommend_portfolio(risk_level: str) -> str:
    """แนะนำพอร์ตการลงทุนตามระดับความเสี่ยง

    Args:
        risk_level: ระดับความเสี่ยงที่รับได้ (low/medium/high)

    Returns:
        แนะนำพอร์ตการลงทุน
    """
    portfolios = {
        "low": "พอร์ตความเสี่ยงต่ำ: 60% พันธบัตร + 30% กองทุนรวมตราสารหนี้ + 10% หุ้นปันผล",
        "medium": "พอร์ตความเสี่ยงปานกลาง: 50% หุ้น + 30% กองทุนรวม + 20% ตราสารหนี้",
        "high": "พอร์ตความเสี่ยงสูง: 80% หุ้น + 15% กองทุนรวม + 5% ทองคำ"
    }
    result = portfolios.get(risk_level.lower(), "กรุณาระบุระดับความเสี่ยง: low, medium, หรือ high")
    logger.info(f"Portfolio recommendation for risk level: {risk_level}")
    return result

tools = [
    calculate_investment_return,
    search_investment_info,
    recommend_portfolio
]
logger.info("Tools created: calculate_investment_return, search_investment_info, recommend_portfolio")

# =============================================================================
# 5. CREATE AGENT
# =============================================================================
# System prompt for investment advisor
system_prompt = """
คุณเป็นที่ปรึกษาการลงทุนที่มีประสบการณ์

ข้อกำหนด:
1. ตอบเฉพาะคำถามที่เกี่ยวกับการลงทุนเท่านั้น
2. ถ้าถามคำถามที่ไม่เกี่ยวกับการลงทุน ให้ตอบว่า "ขออภัย ฉันเป็นที่ปรึกษาการลงทุน สามารถช่วยตอบคำถามเกี่ยวกับการลงทุนเท่านั้น"
3. ใช้ tools ที่มีให้เพื่อคำนวณและค้นหาข้อมูล
4. ตอบเป็นภาษาไทย
5. ถ้าต้องการคำนวณผลตอบแทน ให้ใช้ calculate_investment_return
6. ถ้าต้องการข้อมูลเกี่ยวกับการลงทุน ให้ใช้ search_investment_info
7. ถ้าต้องการแนะนำพอร์ตการลงทุน ให้ใช้ recommend_portfolio
8. บันทึกการทำงานทุกครั้ง
"""

investment_agent = create_agent(
    llm,
    tools,
    system_prompt=system_prompt,
    checkpointer=InMemorySaver()
)
logger.info("Investment Advisor Agent created")

# =============================================================================
# 6. GRADIO INTERFACE
# =============================================================================
def chat(message: str, history: list = None) -> str:
    """Chat function for Gradio"""
    logger.info(f"User message: {message}")

    # Generate unique thread ID for each conversation
    thread_id = str(uuid.uuid4())

    try:
        response = investment_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            {"configurable": {"thread_id": thread_id}}
        )

        # Get last AI message
        ai_message = ""
        for msg in reversed(response['messages']):
            if hasattr(msg, 'type') and msg.type == 'ai':
                ai_message = msg.content
                break

        logger.info(f"Agent response sent: {ai_message[:100]}...")
        return ai_message

    except Exception as e:
        error_msg = f"เกิดข้อผิดพลาด: {str(e)}"
        logger.error(f"Error: {str(e)}")
        return error_msg

# Create Gradio interface
demo = gr.ChatInterface(
    fn=chat,
    title="AI ที่ปรึกษาการลงทุน",
    description="ถามคำถามเกี่ยวกับการลงทุนได้เลย (หุ้น, กองทุน, พันธบัตร, กองทุนรวม ฯลฯ)",
    examples=[
        ["ถ้าลงทุน 50,000 บาท อัตรา 8% ต่อปี 5 ปี จะได้เท่าไหร่?"],
        ["การลงทุนในกองทุนรวมเหมาะกับใคร?"],
        ["แนะนำพอร์ตการลงทุนสำหรับคนที่รับความเสี่ยงได้ปานกลาง"],
        ["หลักการลงทุนที่ดีมีอะไรบ้าง?"]
    ]
)

# =============================================================================
# 7. MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Starting AI Investment Advisor...")
    print("URL: http://localhost:7860")
    print("Logs: logs/agent_YYYYMMDD.log")
    print("=" * 60)

    demo.launch(server_name="localhost", server_port=7860)
