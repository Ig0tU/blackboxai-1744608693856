import gradio as gr
import os
import aiohttp
import asyncio
from git import Repo, GitCommandError
from pathlib import Path
from datetime import datetime
import shutil
import json
import logging
import re
from typing import Dict, List, Optional, Tuple
import subprocess
import plotly.express as px
import plotly.graph_objects as go
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import speech_recognition as sr
from code_editor import code_editor
from functools import lru_cache
import hashlib
import markdown2
from concurrent.futures import ThreadPoolExecutor
from hdbscan import HDBSCAN
import websockets
from websockets.exceptions import ConnectionClosed

# ========== Configuration ==========
WORKSPACE = Path("/tmp/issue_workspace")
WORKSPACE.mkdir(exist_ok=True)
GITHUB_API = "https://api.github.com/repos"
HF_INFERENCE_API = "https://api-inference.huggingface.co/models"
WEBHOOK_PORT = 8002  # Changed from 8000 to avoid conflict with Gradio
WS_PORT = 8001
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
executor = ThreadPoolExecutor(max_workers=4)

# Load tokens from environment variables
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")

HF_MODELS = {
    "Mistral-8x7B": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "Llama-3-8B": "meta-llama/Meta-Llama-3-8B",
    "CodeLlama-34B": "codellama/CodeLlama-34b-Instruct-hf",
    "StarCoder2": "bigcode/starcoder2-15b"
}
# Default Model
DEFAULT_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# ========== Modern Theme ==========
theme = gr.themes.Base(
    primary_hue="violet",
    secondary_hue="emerald",
    font=["Inter", "ui-sans-serif", "system-ui"]
)

# ========== Enhanced Webhook Handler ==========
class WebhookHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b"GitBot is running")

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        payload = json.loads(self.rfile.read(content_length).decode())
        event = self.headers.get('X-GitHub-Event')
        
        if event == 'issues':
            action = payload.get('action')
            if action in ['opened', 'reopened', 'closed', 'assigned']:
                asyncio.run_coroutine_threadsafe(
                    manager.handle_webhook_event(event, action, payload),
                    asyncio.get_event_loop()
                )
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ok"}).encode())

# ========== AI-Powered Issue Manager ==========
class IssueManager:
    """Manages GitHub issues and AI interactions for resolution."""
    def __init__(self) -> None:
        self.issues: Dict[int, dict] = {}
        self.repo_url: Optional[str] = None
        self.repo: Optional[Repo] = None
        self.current_issue: Optional[int] = None
        self.github_token: Optional[str] = None
        self.hf_token: Optional[str] = None
        self.collaborators: Dict[str, dict] = {}
        self.points: int = 0
        self.severity_rules: Dict[str, List[str]] = {
            "Critical": ["critical", "urgent", "security", "crash"],
            "High": ["high", "important", "error", "regression"],
            "Medium": ["medium", "bug", "performance"],
            "Low": ["low", "documentation", "enhancement"]
        }
        self.issue_clusters: Dict[int, List[int]] = {}
        self._init_local_models()
        self.ws_clients: List[websockets.WebSocketClientProtocol] = []
        self.code_editors: Dict[int, dict] = {}
    
    def _init_local_models(self):
        # Skip model initialization when running without deep learning frameworks
        self.code_model = None
        self.summarizer = None
    
    @lru_cache(maxsize=100)
    async def cached_suggestion(self, issue_hash: str, model: str):
        return await self.suggest_resolution(issue_hash, model)

    async def handle_webhook_event(self, event: str, action: str, payload: dict) -> None:
        logger.info(f"Received webhook event: {event} with action: {action}")
        if action == 'closed':
            self.issues.pop(payload['issue']['number'], None)
        else:
            await self.crawl_issues(self.repo_url, self.github_token, self.hf_token)

    async def crawl_issues(self, repo_url: str, github_token: str, hf_token: str) -> Tuple[bool, str]:
        try:
            self.repo_url = repo_url
            self.github_token = github_token
            self.hf_token = hf_token
            
            # Fetch issues using API with proper authentication
            headers = {
                "Authorization": f"token {github_token}" if github_token else "",
                "Accept": "application/vnd.github.v3+json"
            }
            
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(f"{GITHUB_API}/{repo_url}/issues?state=all&per_page=100") as response:
                    if response.status != 200:
                        try:
                            error_data = await response.json()
                            error_message = error_data.get('message', 'Unknown error')
                        except Exception:
                            error_message = 'Unknown error'
                        return False, f"GitHub API error: {error_message}"
                    
                    issues = await response.json()
                    if not isinstance(issues, list):
                        return False, "Invalid response format from GitHub API"
                    
                    # Clear existing issues and store new ones
                    self.issues.clear()
                    for issue in issues:
                        if isinstance(issue, dict) and 'number' in issue:
                            self.issues[issue['number']] = issue
            await self._cluster_similar_issues()
            logger.info(f"Successfully crawled issues: {len(self.issues)} found and clustered into {len(self.issue_clusters)} groups")
            return True, f"Found {len(self.issues)} issues (clustered into {len(self.issue_clusters)} groups)"
        except Exception as e:
            logger.error(f"Error while crawling issues: {e}")
            return False, str(e)

    async def _cluster_similar_issues(self):
        embeddings = await self._generate_embeddings()
        clusterer = HDBSCAN(min_cluster_size=2, metric='cosine')
        clusters = clusterer.fit_predict(embeddings)
        self.issue_clusters = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in self.issue_clusters:
                self.issue_clusters[cluster_id] = []
            self.issue_clusters[cluster_id].append(i)

    async def _generate_embeddings(self):
        async with aiohttp.ClientSession() as session:
            texts = [f"{i['title']} {i['body']}" for i in self.issues.values()]
            response = await session.post(
                f"{HF_INFERENCE_API}/sentence-transformers/all-mpnet-base-v2",
                headers={"Authorization": f"Bearer {self.hf_token}"},
                json={"inputs": texts}
            )
            return await response.json()

    async def generate_code_patch(self, issue_number: int) -> dict:
        issue = self.issues[issue_number]
        context = await self._get_code_context(issue_number)
        prompt = f"""<issue>
{issue['title']}
{issue['body']}
</issue>

<code_context>
{context}
</code_context>

Generate a JSON patch file with specific changes needed to resolve this issue."""
        
        response = self.code_model(
            prompt,
            max_length=1024,
            temperature=0.2,
            return_full_text=False
        )
        try:
            return json.loads(response[0]['generated_text'])
        except json.JSONDecodeError:
            return {"error": "Failed to parse AI-generated patch"}

    async def _get_code_context(self, issue_number: int) -> str:
        repo_path = WORKSPACE / f"repo-{issue_number}"
        code_files = list(repo_path.glob('**/*.py')) + list(repo_path.glob('**/*.js'))
        return "\n".join(f.read_text()[:1000] for f in code_files[:5])

    async def suggest_resolution(self, issue_hash: str, model: str) -> str:
        issue = self.issues[int(issue_hash)]
        prompt = f"""
        ## Issue: {issue['title']}

        {issue['body']}

        Suggest a solution to this issue. 
        """
        
        async with aiohttp.ClientSession() as session:
            response = await session.post(
                f"{HF_INFERENCE_API}/{model}",
                headers={"Authorization": f"Bearer {self.hf_token}"},
                json={"inputs": prompt}
            )
            return await response.json()

    async def broadcast_collaboration_status(self):
        while True:
            try:
                await asyncio.sleep(1)
                await asyncio.gather(
                    *[client.send(json.dumps([{"name": name, "status": status} for name, status in self.collaborators.items()])) for client in self.ws_clients]
                )
            except ConnectionClosed:
                pass

    async def handle_code_editor_update(self, issue_num: int, content: str):
        if issue_num not in self.code_editors:
            self.code_editors[issue_num] = {"content": ""}
        self.code_editors[issue_num]["content"] = content
        await asyncio.gather(
            *[client.send(json.dumps({"type": "code_update", "issue_num": issue_num, "content": content})) for client in self.ws_clients]
        )

# ========== Enhanced UI Components ==========
def create_ui() -> gr.Blocks:
    with gr.Blocks(theme=theme, title="🤖 AI Issue Resolver Pro", css=".gradio-container {max-width: 1200px !important}") as app:
        gr.Markdown("""
        # 🚀 AI Issue Resolver Pro
        *Next-generation issue resolution powered by AI collaboration*
        """)
        
        with gr.Row(variant="panel"):
            with gr.Column(scale=2):
                repo_url = gr.Textbox(label="GitHub Repo", placeholder="https://github.com/org/repo", info="Connect your repository")
                github_token = gr.Textbox(label="GitHub Token", type="password")
                hf_token = gr.Textbox(label="HF Token", type="password")
            
            with gr.Column(scale=1):
                model_select = gr.Dropdown(choices=list(HF_MODELS.keys()), value="Mistral-8x7B", 
                                        label="AI Model", info="Choose your resolution strategy")
                language_select = gr.Dropdown(choices=["python", "javascript", "java", "c", "cpp", "html", "css", "bash", "ruby", "go", "php", "rust", "typescript"], 
                                               value="python", label="Select Language", info="Choose the programming language for the code editor")
                crawl_btn = gr.Button("🚀 Scan Repository", variant="primary")

        with gr.Tabs():
            with gr.Tab("📋 Issue Board", id="board"):
                with gr.Row():
                    issue_list = gr.Dataframe(
                        headers=["ID", "Title", "Severity", "Cluster"],
                        type="array",
                        interactive=True
                    )
                    with gr.Column(scale=1):
                        stats_plot = gr.Plot()
                        collab_status = gr.HTML("<h3>👥 Active Collaborators</h3><div id='collab-list'></div>")

            with gr.Tab("💻 Resolution Studio", id="studio"):
                with gr.Row():
                    with gr.Column(scale=1):
                        issue_num = gr.Number(label="Issue #", precision=0)
                        issue_viz = gr.HTML()
                        ai_tools = gr.Accordion("🛠️ AI Tools")
                        with ai_tools:
                            suggest_btn = gr.Button("🧠 Suggest Resolution")
                            patch_btn = gr.Button("📝 Generate Patch")
                            test_btn = gr.Button("🧪 Create Tests")
                            impact_btn = gr.Button("📊 Impact Analysis")
                        
                    with gr.Column(scale=2):
                        with gr.Tabs():
                            with gr.Tab("Code Editor"):
                                code_edit = gr.HTML(elem_id="code-editor-container")
                            with gr.Tab("AI Chat"):
                                chat = gr.Textbox(
                                    label="AI Suggestions",
                                    interactive=False
                                )

            with gr.Tab("📈 Analytics", id="analytics"):
                with gr.Row():
                    gr.Markdown("### 📅 Resolution Timeline")
                    timeline = gr.Dataframe(
                        headers=["Date", "Event", "Status"],
                        type="array",
                        interactive=False
                    )
                with gr.Row():
                    gr.Markdown("### 🏆 Achievement System")
                    badges = gr.HTML("<div class='badges'>Achievements will be displayed here</div>")
        
        # Event Handlers
        async def generate_patch(issue_num):
            patch = await manager.generate_code_patch(issue_num)
            return gr.JSON(value=patch)

        def update_code_editor(files):
            return code_editor(value=files, language=language_select.value)

        def handle_issue_select(evt):
            if evt and len(evt) > 0:
                issue_id = evt[0]
                body = manager.issues[issue_id]['body'] if issue_id in manager.issues else ""
                return issue_id, body
            return None, ""

        issue_list.select(
            fn=handle_issue_select,
            outputs=[issue_num, issue_viz]
        )

        def handle_crawl(repo, token, hf_token):
            # Clean and parse repository URL
            clean_url = repo.replace("https://github.com/", "").strip("/")
            if "/" not in clean_url:
                return [], None

            try:
                # Create a new event loop in this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Run the async crawl_issues function
                success, message = loop.run_until_complete(manager.crawl_issues(clean_url, token, hf_token))
                
                if success:
                    # Create a plot showing issue statistics
                    stats_data = {
                        "Critical": len([i for i in manager.issues.values() if isinstance(i, dict) and any(label.lower() == "critical" for label in i.get("labels", []))]),
                        "High": len([i for i in manager.issues.values() if isinstance(i, dict) and any(label.lower() == "high" for label in i.get("labels", []))]),
                        "Medium": len([i for i in manager.issues.values() if isinstance(i, dict) and any(label.lower() == "medium" for label in i.get("labels", []))]),
                        "Low": len([i for i in manager.issues.values() if isinstance(i, dict) and any(label.lower() == "low" for label in i.get("labels", []))])
                    }
                    fig = go.Figure(data=[go.Bar(x=list(stats_data.keys()), y=list(stats_data.values()))])
                    fig.update_layout(title="Issue Severity Distribution")
                    
                    # Prepare issue list data
                    issues_data = []
                    for num, issue in manager.issues.items():
                        if not isinstance(issue, dict):
                            continue
                        severity = next((sev for sev, terms in manager.severity_rules.items() 
                                       if any(term in issue.get("title", "").lower() for term in terms)), "Medium")
                        cluster = next((cid for cid, issues in manager.issue_clusters.items() 
                                      if num in issues), -1)
                        issues_data.append([num, issue.get("title", ""), severity, cluster])
                    
                    return issues_data, fig
                else:
                    print(f"Failed to crawl issues: {message}")
                    return [], None
            except Exception as e:
                print(f"Error in handle_crawl: {e}")
                return [], None
            finally:
                loop.close()

        crawl_btn.click(
            fn=handle_crawl,
            inputs=[repo_url, github_token, hf_token],
            outputs=[issue_list, stats_plot],
            api_name=False,
            scroll_to_output=True
        )

        suggest_btn.click(
            fn=lambda issue, model: manager.cached_suggestion(issue, model),
            inputs=[issue_num, model_select],
            outputs=chat
        )

        patch_btn.click(
            fn=generate_patch,
            inputs=[issue_num],
            outputs=chat
        )

        issue_num.change(
            fn=lambda issue_num: create_code_editor(issue_num, language_select.value),
            inputs=[issue_num, language_select],
            outputs=code_edit
        )

    return app

async def handle_ws_connection(websocket):
    manager.ws_clients.append(websocket)
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                if data.get("type") == "code_update":
                    await manager.handle_code_editor_update(data["issue_num"], data.get("content", ""))
            except json.JSONDecodeError:
                print(f"Invalid JSON message received: {message}")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        manager.ws_clients.remove(websocket)

# ========== Execution ==========
if __name__ == "__main__":
    print("Initializing GitBot...")
    manager = IssueManager()
    app = create_ui()

    async def start_servers():
        """Start WebSocket server and collaboration broadcast"""
        print(f"Starting WebSocket server on port {WS_PORT}...")
        ws_server = await websockets.serve(handle_ws_connection, "localhost", WS_PORT)
        
        # Start collaboration status broadcast
        broadcast_task = asyncio.create_task(manager.broadcast_collaboration_status())
        
        try:
            # Keep the servers running
            await asyncio.gather(ws_server.wait_closed(), broadcast_task)
        except Exception as e:
            print(f"Error in WebSocket server: {e}")

    # Start webhook server
    print(f"Starting webhook server on port {WEBHOOK_PORT}...")
    webhook_server = HTTPServer(("", WEBHOOK_PORT), WebhookHandler)
    webhook_thread = threading.Thread(target=webhook_server.serve_forever, daemon=True)
    webhook_thread.start()

    # Initialize event loop and start WebSocket server
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    ws_thread = threading.Thread(
        target=lambda: asyncio.run(start_servers()),
        daemon=True
    )
    ws_thread.start()

    # Start Gradio interface
    print("Starting Gradio interface...")
    try:
        app.launch(
            server_name="0.0.0.0",
            server_port=8000,
            share=False
        )
        print("Gradio interface is running at http://0.0.0.0:8000")
    except Exception as e:
        print(f"Error starting Gradio interface: {e}")
