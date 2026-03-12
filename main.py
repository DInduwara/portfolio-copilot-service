from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import json
import os

load_dotenv()

app = FastAPI(title="Dinuka's AI Twin API")

# Ensure your Vercel URL is here with NO trailing slash!
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "https://dinuka-induwara-portfolio.vercel.app" 
    ], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@tool
def control_ui(action: str, target: str):
    """Controls the user's frontend UI and navigation.
    Allowed actions: 
    - 'scroll': For sections on the HOME page (targets: 'projects', 'about', 'tech').
    - 'navigate': To change the actual page URL.
    - 'download': To trigger a file download to the user's device.
    
    Allowed targets for 'navigate': 
    - '/projects': The dedicated Projects Gallery page.
    - '/projects/[slug]': Specific project pages (e.g., '/projects/friday-ai-platform').
    - '/contact': The Contact page.
    
    Allowed targets for 'download':
    - 'resume': Dinuka's CV.
    """
    return f"UI_COMMAND::{action}::{target}"

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, streaming=True) 
memory = MemorySaver()
agent_executor = create_react_agent(llm, tools=[control_ui], checkpointer=memory)

class ChatRequest(BaseModel):
    message: str
    session_id: str
    context: str 

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    
    # SENIOR PROMPT ENGINEERING APPLIED HERE
    DYNAMIC_SYSTEM_PROMPT = f"""
    You are the AI Digital Twin of Dinuka Induwara. You speak on his behalf as a friendly, passionate, and professional software engineer. 
    You adopt his persona, using "I" and "my" when talking about his experience and projects, but gracefully acknowledge you are his AI assistant if explicitly asked.

    ### CONVERSATION STYLE (CRITICAL RULES):
    1. BE HUMAN & NATURAL: NEVER just regurgitate a bulleted list of data, skills, or projects. Weave the provided information into natural, conversational paragraphs.
    2. THE "ELEVATOR PITCH": If asked "Who are you?", "Tell me about yourself", or "Tell me about him", give a concise 2-3 sentence summary. (e.g., "I'm a BEng Software Engineering undergraduate at IIT and an Intern Software Engineer. I'm deeply focused on AI engineering, building multi-agent systems, and full-stack development...").
    3. KEEP IT BRIEF & ENGAGING: Do not overwhelm the user. Give short, punchy answers. Always end your response by asking a relevant guiding question to keep the chat moving (e.g., "Would you like to hear about my multi-agent AI projects or check out my CV?").
    4. NO BULLET POINTS: Unless the user explicitly asks for a "list", use paragraph form.

    ### DINUKA'S CONTEXT DATA:
    {request.context}
    
    ### NAVIGATION & SCROLLING RULES (STRICT):
    1. If asked to "Show featured projects" or "Scroll to projects":
       - Use action='scroll', target='projects'.
    2. If asked for "All projects" or "Show your projects" (generally):
       - Use action='navigate', target='/projects'.
    3. If asked about a SPECIFIC project (e.g. FRIDAY or IntelliRAG):
       - Use action='navigate', target='/projects/[slug]'.
    4. If asked for his resume or CV:
       - Use action='download', target='resume'.
       - IMPORTANT: Tell the user exactly this: "I've triggered the download for my CV. Check your downloads folder!" Do NOT say you opened it or are showing it.
    """
    
    system_msg = SystemMessage(content=DYNAMIC_SYSTEM_PROMPT)
    user_msg = HumanMessage(content=request.message)

    async def event_stream():
        config = {"configurable": {"thread_id": request.session_id}}
        
        async for msg, metadata in agent_executor.astream(
            {"messages": [system_msg, user_msg]},
            config=config,
            stream_mode="messages"
        ):
            if metadata.get("langgraph_node") == "agent" and msg.content:
                yield f"data: {json.dumps({'chunk': msg.content})}\n\n"
            
            elif metadata.get("langgraph_node") == "tools":
                if isinstance(msg.content, str) and "UI_COMMAND::" in msg.content:
                    _, action, target = msg.content.split("::")
                    yield f"data: {json.dumps({'command': {'action': action, 'target': target}})}\n\n"
        
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/")
def read_root():
    return {"status": "AI Twin Online"}