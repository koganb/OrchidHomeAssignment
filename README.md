# Orchid Home Assignment

Repository Intelligence is a local FastAPI and Streamlit application for indexing GitHub repositories and asking code-focused questions about them. It currently supports Python and Java repositories.

The indexing pipeline clones a repository, extracts symbols with Tree-sitter, builds a NetworkX call graph, stores symbol embeddings in ChromaDB, and uses a LangChain ReAct agent with OpenAI to answer repository questions.

## Architecture

The application has three main layers:

- Streamlit UI: collects the GitHub repository URL, shows processing progress, and renders the repository chat interface.
- FastAPI backend: owns repository processing, background job status, cached manifests, and chat requests.
- Indexing and chat pipeline: downloads repositories, extracts symbols, builds a call graph, persists embeddings, and gives the chat agent tools for semantic code lookup and execution-flow tracing.

Typical request flow:

1. The user submits a GitHub URL in Streamlit.
2. Streamlit calls `POST /process/start`.
3. FastAPI starts a background processing job.
4. The processing pipeline clones or reuses the repository, extracts symbols, builds a graph, and stores embeddings.
5. Streamlit polls `GET /process/status/{job_id}` until processing completes.
6. The user asks a repository question.
7. Streamlit calls `POST /chat` with the ChromaDB path and NetworkX graph path.
8. The LangChain agent uses the embedded symbol store and call graph tools to answer the question.

## Requirements

- Python 3.12 or newer
- `uv`
- `git` available on `PATH`
- OpenAI API key

## Setup

Install dependencies:

```powershell
uv sync
```

Create a local environment file:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and set:

```text
OPENAI_API_KEY=your_openai_api_key
```

LangSmith tracing is optional. The example file includes the supported LangSmith variables if you want tracing enabled.

## Run The App

Start both the FastAPI backend and the Streamlit UI:

```powershell
uv run python run_app.py
```

Then open the Streamlit URL printed in the terminal, usually:

```text
http://localhost:8501
```

The API runs on:

```text
http://127.0.0.1:8000
```

If the Streamlit UI needs to point at a different API host, set:

```powershell
$env:REPOSITORY_API_BASE_URL="http://127.0.0.1:8000"
```

## Usage

1. Paste a GitHub repository URL into the Streamlit app.
2. Wait for the repository to be processed.
3. Ask questions about repository code, structure, features, or execution flow.

The chat endpoint intentionally rejects non-repository questions.

## API

Health check:

```http
GET /health
```

Process a repository synchronously:

```http
POST /process
```

Payload:

```json
{
  "github_url": "https://github.com/owner/repo",
  "force": false
}
```

Start repository processing in the background:

```http
POST /process/start
```

Check background processing status:

```http
GET /process/status/{job_id}
```

Ask a question about a processed repository:

```http
POST /chat
```

Payload:

```json
{
  "question": "Explain the main request flow.",
  "chroma_path": "chroma/owner-repo",
  "networkx_path": "graphs/owner-repo.pkl",
  "collection_name": "symbol_embeddings"
}
```

## Generated Data

The app writes generated repository artifacts locally:

- `processed_repositories/` stores cloned repositories and manifests.
- `chroma/` stores persistent ChromaDB collections.
- `graphs/` stores persisted NetworkX call graphs.

Use `force: true` on `/process` to reprocess a repository and refresh cached output.

## Tests

Run the test suite:

```powershell
uv run pytest
```

One integration test processes a public GitHub repository and calls the chat endpoint, so it requires network access and a valid `OPENAI_API_KEY`.

## Known Limitations

- No chat memory: each question is answered independently.
- The repository-question guardrails are too aggressive and may reject valid repository questions.

## Project Structure

```text
api.py                    FastAPI service and repository processing endpoints
run_app.py                Starts FastAPI and Streamlit together
chat/                     LangChain agent, tools, and request policy
data_preparation/         Repository download, symbol extraction, embeddings, and graph building
streamlit_gui/            Streamlit UI and API client
tests/                    API and embedding tests
```
