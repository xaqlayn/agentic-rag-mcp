# Company knowledge MCP server (for the Unsloth Studio chat)

An MCP server that gives the model in your Unsloth chat four tools. The chat model is the agent: it reads the
tool descriptions, decides which tool to call, reads the result, and repeats until it can answer.
This project only provides the tools - there is no agent loop and no chat API in it.

```
 Unsloth Studio chat (LLM decides)  ──MCP over HTTP──►  this server
                                                         ├─ search_docs      hybrid keyword + semantic search  → Qdrant
                                                         ├─ query_database   read-only SQL on your CSV tables → SQLite
                                                         ├─ calculator       exact arithmetic
                                                         └─ get_current_time
```

| Part | Runs on | Used for |
|---|---|---|
| LLM + chat UI | Unsloth Studio (outside this project) | decides which tool to call, writes the answer |
| MCP server | container `mcp` (this code) | exposes the tools at `http://<host>:8000/mcp` |
| Embeddings | `llama-server --embeddings` (container `embeddings`, CPU) | text → vectors |
| Vector DB | Qdrant (container `qdrant`) | dense + BM25-style sparse vectors, fused with RRF |

## Files

```
app/server.py        the MCP server: tool registration, bearer-token check, /health   (python -m app.server)
app/tools.py         tool logic (search, SQL, calculator, time)
app/vectorstore.py   Qdrant collection, upsert, hybrid search
app/embeddings.py    llama-server embedding client
app/tables.py        CSV -> SQLite, read-only SQL with safety checks
app/textproc.py      chunking, tokenizer, sparse (keyword) vectors
app/ingest.py        loads documents and tables      (python -m app.ingest)
app/check.py         tests every connection          (python -m app.check)
app/config.py        all settings (env vars / .env)
SYSTEM_PROMPT.md     rules to paste into the Unsloth chat's system prompt
data/docs/           your .txt / .md files  (sample files included)
data/tables/         your .csv files        (sample files included)
tests/               offline tests (no servers needed)
```

## Setup

Run these on the Ubuntu server (over SSH, no desktop needed):

1. **Install Docker** (skip this if `docker compose version` already works):
   ```bash
   curl -fsSL https://get.docker.com | sudo sh
   sudo usermod -aG docker $USER        # then log out and back in once
   ```
   If your company does not allow piping a script into a shell, follow Docker's official Ubuntu install guide instead.
2. **Copy the project to the server.** On your own computer: `scp agentic-rag-mcp-v3.zip <user>@<server>:~/`.
   On the server:
   ```bash
   sudo apt install -y unzip
   unzip agentic-rag-mcp-v3.zip && cd agentic-rag-mcp
   ```
3. **Download an embedding model** into `models/` (the server needs internet access for this one step):
   ```bash
   mkdir -p models
   # multilingual (German + English), 634 MB - recommended for German documents:
   wget -c -P models https://huggingface.co/ggml-org/bge-m3-Q8_0-GGUF/resolve/main/bge-m3-q8_0.gguf
   # English-focused, smaller and faster (needs the prefixes described in .env.example):
   # wget -c -P models https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q8_0.gguf
   ```
   No internet on the server? Download the file on your own computer and `scp` it into `models/`.
4. **Config.** `cp .env.example .env`, then `nano .env`. Set `MCP_AUTH_TOKEN` to a long random secret
   (`python3 -c "import secrets; print(secrets.token_urlsafe(32))"`). If you chose another model, set `EMBED_GGUF`
   to its file name (and the prefixes, if the model needs them).
5. **Start the containers and check.**
   ```bash
   docker compose up -d --build
   docker compose ps                                # qdrant, embeddings, mcp should be running
   docker compose exec mcp python -m app.ingest     # load documents + tables
   docker compose exec mcp python -m app.check      # every line must say PASS
   ```
   The containers restart automatically after a reboot of the server.
   After changing CSV files run `docker compose restart mcp` (the table schema is read at startup).

Without Docker: `pip install -r requirements.txt`, run Qdrant and llama-server yourself, set the URLs in `.env`,
then `python -m app.ingest` and `python -m app.server`.

## Connect Unsloth Studio

Following [Unsloth's MCP guide](https://unsloth.ai/docs/basics/mcp):

1. In the chat toolbar click **MCP**, then **Add custom MCP**.
2. **Display name:** `company-knowledge`. **URL:** `http://<machine running the container>:8000/mcp`.
   The URL has to be reachable from where Studio runs; `localhost` only works if both are on the same machine.
3. Leave **OAuth** off and add a custom header: `Authorization` = `Bearer <your MCP_AUTH_TOKEN>`.
4. **Test connection**, then **Add server**. It should report 4 tools (click **Refresh** if not).
5. Switch on the server's toggle and the **Use MCP Servers** master toggle, pick a model, and chat.
6. Optional but recommended for local models: paste the rules from `SYSTEM_PROMPT.md` into the chat's system prompt.

If *Test connection* fails, run `docker compose exec mcp python -m app.check`: it tells you whether the server is
down, the token is wrong (HTTP 401), or the path is wrong (HTTP 404). If Studio insists on a bare base URL, set
`MCP_PATH=/` in `.env`, restart, and use `http://<machine>:8000/`. Clients that only speak SSE: start the server
with `--transport sse` (endpoint `/sse`). Newer Studio versions can also launch a stdio server themselves
(`python -m app.server --transport stdio`), but then Python and this project must be on the machine where Studio runs.

Questions to try with the sample data:
- "How long do I have to take carried-over vacation?" (documents)
- "Which region had the highest revenue?" (database)
- "How many hours of leave do new employees get per year, assuming 8h days?" (documents + calculator)
- "How many days since the last order in the orders table?" (database + time + calculator)

## Your own data

- **Documents:** put `.txt` or `.md` files into `data/docs/` (subfolders are fine) and run `python -m app.ingest`.
  Re-running is safe: chunks get stable IDs and are overwritten. Changed the embedding model? Add `--recreate`.
  PDFs are optional: uncomment `pypdf` in `requirements.txt`.
- **Tables:** put `.csv` files into `data/tables/`. Each file becomes a table (file name = table name). Column types
  are detected automatically; `;` and `,` separators and UTF-8 / Excel (cp1252) encodings work. Use a dot as the
  decimal separator (`12.5`, not `12,5`), otherwise the column is read as text. The model sees the table and column
  names (plus one example row) in the `query_database` tool description.

## How the model decides

What the model knows about your tools is exactly what the server sends: the tool names, the descriptions and
parameter descriptions in `app/server.py`, and the short `INSTRUCTIONS` text. This is where you tune behaviour: if
the model picks the wrong tool, make the descriptions more specific. Adding a tool means writing a function in
`app/tools.py` and one `@mcp.tool(...)` block in `app/server.py`.

The tool-calling loop, step limits and conversation memory now belong to the Unsloth chat, not to this project.

## Things to know

- **Speed:** the embedding container runs on the CPU. That is fine for the sample data, but a large document set (around 1 GB of text) can take many hours to index. Time `app.ingest` on a small batch of your real documents first and scale up. A smaller model (nomic-embed) is much faster than bge-m3. GPU builds of llama.cpp's server image exist, but they need extra setup on AMD hardware.
- **Network:** port 8000 is published on all interfaces and protected by the token. If Studio runs on this same server, change `"8000:8000"` in `docker-compose.yml` to `"127.0.0.1:8000:8000"` so only this machine can reach it. Qdrant and the embedding server are not reachable from outside the server at all.
- **The chat model must support tool calling.** Small models sometimes choose the wrong tool or write wrong SQL;
  the error text goes back to the model so it can retry. A bigger model helps most.
- **SQL safety:** only `SELECT`/`WITH`, read-only database file, an authorizer that blocks everything but reads,
  row and time limits. Give the server only tables that every user may see.
- **Access rights:** there are none per user. Everyone who can use the Studio chat with this server sees all indexed
  documents and tables. The bearer token only keeps unauthorized programs out.
- **Prompt injection:** text from your documents goes to the model. Index only documents you trust, and keep Studio's
  tool-call confirmation on if you are unsure.
- **Production:** pin the Qdrant image version, and consider a reranker and testing chunk sizes on your real documents.

## Tests

```bash
pip install pytest
pytest -q        # 39 tests, no servers needed
```
They run the real MCP client against the server, in-process and over HTTP (including the token check), with
in-memory Qdrant and fake embeddings.
