# RAG Evaluator

A lightweight, self-hosted RAG evaluation service written in Go. It scores RAG pipeline outputs in real time using embedding-based metrics and an optional LLM-as-a-judge layer, with a built-in web dashboard.

## Overview

`rageval` sits alongside your RAG pipelines (Flogo, BusinessWorks 6, LangChain, or any custom platform) and receives eval events after each query. Scoring is fully asynchronous — the event endpoint returns immediately so your pipeline is never blocked. Results are stored in SQLite and surfaced via a dashboard and REST API.

```
RAG Pipeline  ──POST /eval/v1/events──▶  rageval
                                            │
                                    async worker pool
                                            │
                                    ┌───────▼────────┐
                                    │  Embed Provider │  (Ollama)
                                    │  LLM Judge      │  (Ollama, optional)
                                    └───────┬────────┘
                                            │
                                       SQLite store
                                            │
                                    Dashboard / REST API
```

## Metrics

### Embedding-based

| Metric | Method | Description |
|---|---|---|
| **Context Relevance** | Embedding cosine | Mean cosine similarity between the query and each retrieved chunk |
| **Context Precision@K** | Embedding cosine | Weighted relevance of top-K chunks; higher-ranked relevant chunks score more |
| **Faithfulness** | Embedding cosine | % of answer sentences whose embedding is grounded in the chunk corpus |
| **Answer Relevance** | Embedding cosine | Cosine similarity between query and final answer |
| **Overall Score** | Weighted average | ctx_relevance×0.25 + ctx_precision×0.10 + faithfulness×0.35 + answer_relevance×0.20 + chunk_utilization×0.10 |

### LLM judge (optional, requires `llmJudge.enabled: true`)

| Metric | Description |
|---|---|
| **LLM Context Relevance** | Rubric-based 1–5 score — how well the retrieved chunks address the query |
| **LLM Faithfulness** | Rubric-based 1–5 score — how well the answer is grounded in the chunks |
| **LLM Claim Faithfulness** | Atomic fact check — fraction of answer claims that can be verified in the retrieved chunks |
| **LLM Answer Relevance** | Rubric-based 1–5 score — how well the answer addresses the query |
| **LLM Overall** | Weighted average: context_relevance×0.40 + faithfulness×0.35 + answer_relevance×0.25 |

## Prerequisites

- [Ollama](https://ollama.com) running locally with the embedding model pulled:
  ```sh
  ollama pull nomic-embed-text
  ```
- For the LLM judge, also pull the judge model:
  ```sh
  ollama pull qwen2.5:14b
  ```

## Quick Start

### Run locally

```sh
# Clone and build
git clone https://github.com/mpandav-tibco/rageval
cd rageval
go build -o rageval .

# Edit config.yaml to match your environment, then start
bash start.sh
```

The service starts on the configured port (default `9090`) in the background. Logs are written to `rageval-svc.log`.

Verify it is running:
```sh
curl http://localhost:9090/health
# {"status":"ok","time":"..."}
```

Open the dashboard at **http://localhost:9090/**.

<img width="2534" height="911" alt="image" src="https://github.com/user-attachments/assets/bfc62e85-a628-4b99-9523-fc003ff07f56" />



### Run with Docker Compose

Requires [Docker](https://docs.docker.com/get-docker/) and Ollama running on the host.

```sh
# Pull required Ollama models on the host first
ollama pull nomic-embed-text
ollama pull qwen2.5:14b

# Start rageval (builds image if not already built)
docker compose up -d
```

The compose file uses `config.docker.yaml` which points Ollama at `host.docker.internal:11434` and stores the SQLite database in a named volume at `/data/rageval.db`. On Linux, the `extra_hosts: host-gateway` entry in `docker-compose.yml` makes the host reachable at `host.docker.internal`; on Docker Desktop (Mac/Windows) this works automatically.

Verify:
```sh
curl http://localhost:9090/health
```

To rebuild after code changes:
```sh
docker compose up -d --build
```

To view logs:
```sh
docker compose logs -f rageval
```

## Configuration

Configuration is loaded from `config.yaml` (or the path set in the `RAGEVAL_CONFIG` environment variable).

```yaml
server:
  port: 9090
  logLevel: info  # info | debug

embed:
  provider: ollama               # only "ollama" is supported today
  url: http://localhost:11434
  model: nomic-embed-text
  batchSize: 32
  timeoutSeconds: 30

storage:
  type: sqlite                   # only "sqlite" is supported today
  path: /path/to/rageval.db

eval:
  workerCount: 2                 # parallel scoring workers
  channelBuffer: 1000            # in-flight event queue depth
  sampleRate: 1.0                # fraction of events to score (1.0 = all)
  faithfulnessThreshold: 0.60    # min cosine score for a sentence to be "grounded"
  aggregationMode: mean          # mean | min | max — how chunk scores are combined
  llmJudge:
    enabled: true
    url: http://localhost:11434
    model: qwen2.5:14b
    timeoutSeconds: 300
```

| Key | Description |
|---|---|
| `server.port` | HTTP listening port |
| `server.logLevel` | Log verbosity: `info` (default) or `debug`. Override with `RAGEVAL_LOG_LEVEL=debug` env var |
| `embed.provider` | Embedding backend (`ollama`) |
| `embed.model` | Model used for all embedding calls |
| `eval.sampleRate` | Set `< 1.0` to evaluate only a fraction of traffic |
| `eval.faithfulnessThreshold` | Cosine threshold below which an answer sentence is flagged as ungrounded |
| `eval.aggregationMode` | How per-chunk scores are combined: `mean` (default), `min`, or `max` |
| `eval.llmJudge.enabled` | Toggle the LLM-as-a-judge scoring layer |
| `eval.llmJudge.model` | Ollama model for LLM judge (default `qwen2.5:14b`) |
| `eval.llmJudge.timeoutSeconds` | Per-query LLM judge timeout in seconds (default `300`) |

## API Reference

### `POST /eval/v1/events`

Submit a RAG evaluation event. Returns `202 Accepted` immediately — scoring happens asynchronously in the background and never blocks the caller's pipeline. Events with no chunks or no query field are silently discarded (also `202`).

Accepts **JSON** (Flogo / curl) and **XML** (BW HTTP Client with `ragevalEvent` XSD).

**JSON body**

```json
{
  "pipelineId": "my-pipeline",
  "platform": "flogo",
  "traceId": "trace-abc-123",
  "collection": "product-docs",
  "query": "What is the return policy?",
  "chunks": [
    { "id": "chunk-1", "text": "Returns are accepted within 30 days.", "score": 0.91 },
    { "id": "chunk-2", "text": "Items must be in original packaging.", "score": 0.87 }
  ],
  "answer": "You can return items within 30 days if they are in original packaging.",
  "expectedAnswer": "Returns accepted within 30 days in original packaging."
}
```

| Field | Required | Description |
|---|---|---|
| `pipelineId` | No | Identifier for the pipeline (for grouping in dashboard) |
| `platform` | No | `flogo` \| `bw` \| `langchain` \| `custom` |
| `traceId` | No | Unique trace ID for correlation |
| `collection` | No | Vector DB collection name |
| `query` | **Yes** | The user's question |
| `chunks` | **Yes*** | Retrieved context chunks (`id`, `text`, `score`) |
| `selectedEmbeddings` | **Yes*** | BW Query activity output — plain `string[]`; synthesised into chunks automatically |
| `answer` | **Yes** | The generated answer to evaluate |
| `expectedAnswer` | No | Ground-truth answer; enables `answerCorrectness` metric |

\* Either `chunks` or `selectedEmbeddings` must be provided.

**Flogo ragQuery `sourceDocuments` format** is also accepted — chunks can carry `content` or `payload.text` instead of `text`.

**XML body** (BW)

```xml
<ragevalEvent>
  <pipelineId>my-pipeline</pipelineId>
  <platform>bw</platform>
  <traceId>trace-abc-123</traceId>
  <collection>product-docs</collection>
  <query>What is the return policy?</query>
  <chunks>
    <chunk><id>chunk-1</id><text>Returns are accepted within 30 days.</text><score>0.91</score></chunk>
  </chunks>
  <answer>You can return items within 30 days.</answer>
</ragevalEvent>
```

---

### `GET /eval/v1/metrics`

Returns aggregated quality metrics.

| Query param | Description |
|---|---|
| `collection` | Filter to a specific collection |
| `platform` | Filter to a specific platform |
| `period` | Rolling window: `7d`, `30d`, or a plain integer (days). Omit for all time. |

---

### `GET /eval/v1/results`

Returns per-query scored rows for the dashboard table.

| Query param | Default | Description |
|---|---|---|
| `collection` | — | Filter by collection |
| `platform` | — | Filter by platform |
| `limit` | `500` | Maximum rows to return |

---

### `GET /eval/v1/platforms`

Returns the list of distinct `platform` values seen so far.

---

### `GET /eval/v1/collections`

Returns the list of distinct `collection` names seen so far.

---

### `DELETE /eval/v1/results`

Deletes all stored eval records. Pass `?collection=<name>` to scope the reset to a single collection.

---

### `GET /health`

```json
{ "status": "ok", "time": "2026-05-06T09:45:20Z" }
```

## Dashboard

A single-page dashboard is embedded in the binary and served at `/`. It shows:

- **LLM judge metric cards**: LLM Overall, LLM Faithfulness, LLM Claim Faithfulness, LLM Answer Relevance, LLM Context Relevance
- **Embedding metric cards**: Overall, Context Relevance, Context Precision@K, Faithfulness, Answer Relevance
- **Diagnostics**: Hallucination %, ungrounded sentence rate
- **Per-query results table** with drill-down modal showing:
  - RAG answer text
  - LLM judge scores per query
  - Embedding-based retrieval diagnostics
  - Judge commentary / reasoning
  - Embedding flags (ungrounded sentences)
- Collection and platform filter dropdowns
- Period filter (7d / 30d / all time)
- Reset button (scoped to selected collection or global)

## Platform Integration

### Flogo

Use the **ragQuery** activity's `sourceDocuments` output directly as `chunks`. The service resolves text from `content` or `payload.text` fields automatically.

### BusinessWorks 6

The BW HTTP Client can POST the `ragevalEvent` XML payload directly. Alternatively, pass `selectedEmbeddings` as a string array from the BW Query activity output.

### curl

```sh
curl -s -X POST http://localhost:9090/eval/v1/events \
  -H "Content-Type: application/json" \
  -d '{
    "pipelineId": "test",
    "platform": "custom",
    "traceId": "t-001",
    "collection": "docs",
    "query": "What is RAG?",
    "chunks": [{"id":"1","text":"RAG stands for Retrieval-Augmented Generation.","score":0.95}],
    "answer": "RAG stands for Retrieval-Augmented Generation."
  }'
```

## Project Structure

```
rageval/
├── main.go                   # Entry point — wires config, storage, engine, HTTP server
├── config.yaml               # Runtime configuration (local)
├── config.docker.yaml        # Runtime configuration for Docker (host.docker.internal URLs, /data DB path)
├── Dockerfile                # Multi-stage build: golang:1.25-alpine → distroless/static
├── docker-compose.yml        # Compose stack: rageval + named volume for SQLite
├── start.sh                  # Background start helper (local)
├── dashboard/
│   └── index.html            # Embedded SPA dashboard
└── internal/
    ├── api/
    │   └── handler.go        # HTTP route handlers; JSON + XML ingestion
    ├── config/
    │   └── config.go         # Config struct + YAML loader
    ├── engine/
    │   ├── embed.go          # Ollama batch embedding provider
    │   ├── scorer.go         # Embedding-based metric computation
    │   └── llm_judge.go      # LLM-as-a-judge layer (Ollama)
    ├── queue/
    │   └── worker.go         # Async worker pool
    └── store/
        ├── store.go          # EvalStore interface
        └── sqlite.go         # SQLite implementation
```

## Building

```sh
go build -o rageval .
```

Requires Go 1.25+.
