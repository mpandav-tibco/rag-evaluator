package api

import (
	"encoding/json"
	"encoding/xml"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/mpandav-tibco/rageval/internal/engine"
	"github.com/mpandav-tibco/rageval/internal/queue"
	"github.com/mpandav-tibco/rageval/internal/store"
)

// EventRequest is the JSON body for POST /eval/v1/events.
// This is the platform-agnostic contract — any pipeline can POST to this.
type EventRequest struct {
	PipelineID string      `json:"pipelineId"`
	Platform   string      `json:"platform"` // flogo | bw | langchain | custom
	TraceID    string      `json:"traceId"`
	Collection string      `json:"collection"`
	Query      string      `json:"query"`
	Chunks     ChunksField `json:"chunks"`
	// SelectedEmbeddings accepts the BW Query activity output format: a plain
	// string array. When chunks is empty/absent, each string is synthesised into
	// a ChunkInput with a sequential id and a default score of 1.0.
	SelectedEmbeddings []string `json:"selectedEmbeddings"`
	Answer             string   `json:"answer"`
	ExpectedAnswer     string   `json:"expectedAnswer"` // optional
}

// ChunkInput accepts both flat {id,text,score} and the Flogo ragQuery
// sourceDocuments format {id,score,content:"<string>",payload:{text:...}}.
type ChunkInput struct {
	ID      string                 `json:"id"`
	Text    string                 `json:"text"` // flat format
	Score   float64                `json:"score"`
	Content string                 `json:"content"` // Flogo ragQuery: extracted content string
	Payload map[string]interface{} `json:"payload"` // Flogo ragQuery: full payload object
}

// ChunksField unmarshals chunks in two JSON shapes:
//   - flat array:        [{"id":...,"text":...,"score":...}, ...]          (Flogo / curl)
//   - BW XSD object:    {"chunk":[{"id":...,"text":...,"score":...}, ...]} (BW HTTP Client)
type ChunksField []ChunkInput

func (cf *ChunksField) UnmarshalJSON(data []byte) error {
	// Try flat array first
	var arr []ChunkInput
	if err := json.Unmarshal(data, &arr); err == nil {
		*cf = arr
		return nil
	}
	// Try BW XSD object shape: {"chunk": [...]}
	var obj struct {
		Chunk []ChunkInput `json:"chunk"`
	}
	if err := json.Unmarshal(data, &obj); err == nil {
		*cf = obj.Chunk
		return nil
	}
	return fmt.Errorf("chunks: cannot parse as array or BW {chunk:[...]} object")
}

// resolveText returns the chunk text from whichever field is populated.
// Priority: text > content (Flogo ragQuery extracted string) > payload.text
func (c ChunkInput) resolveText() string {
	if c.Text != "" {
		return c.Text
	}
	if c.Content != "" {
		return c.Content
	}
	if c.Payload != nil {
		if v, ok := c.Payload["text"]; ok {
			if s, ok := v.(string); ok {
				return s
			}
		}
	}
	return ""
}

// xmlEventRequest mirrors EventRequest for BW XML payloads (ragevalEvent XSD).
type xmlEventRequest struct {
	XMLName    xml.Name      `xml:"ragevalEvent"`
	PipelineID string        `xml:"pipelineId"`
	Platform   string        `xml:"platform"`
	TraceID    string        `xml:"traceId"`
	Collection string        `xml:"collection"`
	Query      string        `xml:"query"`
	Chunks     xmlChunksType `xml:"chunks"`
	Answer     string        `xml:"answer"`
}

type xmlChunksType struct {
	Chunks []xmlChunk `xml:"chunk"`
}

type xmlChunk struct {
	ID    string  `xml:"id"`
	Text  string  `xml:"text"`
	Score float64 `xml:"score"`
}

// Handler holds all HTTP route handlers.
type Handler struct {
	pool  *queue.WorkerPool
	store store.EvalStore
}

func NewHandler(pool *queue.WorkerPool, st store.EvalStore) *Handler {
	return &Handler{pool: pool, store: st}
}

// RegisterRoutes binds all routes to the given mux.
func (h *Handler) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("POST /eval/v1/events", h.handleEvent)
	mux.HandleFunc("GET /eval/v1/metrics", h.handleMetrics)
	mux.HandleFunc("GET /eval/v1/platforms", h.handlePlatforms)
	mux.HandleFunc("GET /eval/v1/results", h.handleResults)
	mux.HandleFunc("GET /eval/v1/collections", h.handleCollections)
	mux.HandleFunc("DELETE /eval/v1/results", h.handleReset)
	mux.HandleFunc("GET /health", h.handleHealth)
}

// handleEvent accepts a RAG eval event and enqueues it for async scoring.
// Returns 202 Accepted immediately — never blocks the caller's pipeline.
// Accepts both JSON (Flogo / curl) and XML (BW HTTP Client with ragevalEvent XSD).
func (h *Handler) handleEvent(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, `{"error":"cannot read body"}`, http.StatusBadRequest)
		return
	}
	slog.Debug("eval event raw body", "contentType", r.Header.Get("Content-Type"), "bodyLen", len(body))

	var req EventRequest
	ct := r.Header.Get("Content-Type")
	if strings.Contains(ct, "xml") || (len(body) > 0 && body[0] == '<') {
		// BW sends XML — parse and map to EventRequest
		var xr xmlEventRequest
		if err := xml.Unmarshal(body, &xr); err != nil {
			slog.Error("invalid XML body", "error", err, "body", string(body))
			http.Error(w, `{"error":"invalid XML"}`, http.StatusBadRequest)
			return
		}
		req.PipelineID = xr.PipelineID
		req.Platform = xr.Platform
		req.TraceID = xr.TraceID
		req.Collection = xr.Collection
		req.Query = xr.Query
		req.Answer = xr.Answer
		for _, c := range xr.Chunks.Chunks {
			req.Chunks = append(req.Chunks, ChunkInput{ID: c.ID, Text: c.Text, Score: c.Score})
		}
	} else {
		if err := json.Unmarshal(body, &req); err != nil {
			slog.Error("invalid JSON body", "error", err, "body", string(body))
			http.Error(w, `{"error":"invalid JSON"}`, http.StatusBadRequest)
			return
		}
	}

	if req.Query == "" || req.Answer == "" {
		http.Error(w, `{"error":"query and answer are required"}`, http.StatusBadRequest)
		return
	}

	// BW Query activity returns selectedEmbeddings ([]string) instead of chunks.
	// Synthesise chunk objects when chunks is absent/empty.
	if len(req.Chunks) == 0 && len(req.SelectedEmbeddings) > 0 {
		for i, text := range req.SelectedEmbeddings {
			req.Chunks = append(req.Chunks, ChunkInput{
				ID:    fmt.Sprintf("bw-chunk-%d", i+1),
				Text:  text,
				Score: 1.0,
			})
		}
	}

	if len(req.Chunks) == 0 {
		http.Error(w, `{"error":"chunks or selectedEmbeddings are required"}`, http.StatusBadRequest)
		return
	}

	// Map to internal request
	chunks := make([]engine.Chunk, len(req.Chunks))
	for i, c := range req.Chunks {
		chunks[i] = engine.Chunk{ID: c.ID, Text: c.resolveText(), Score: c.Score}
	}

	evalReq := engine.EvalRequest{
		PipelineID:     req.PipelineID,
		Platform:       req.Platform,
		TraceID:        req.TraceID,
		Collection:     req.Collection,
		Query:          req.Query,
		Chunks:         chunks,
		Answer:         req.Answer,
		ExpectedAnswer: req.ExpectedAnswer,
	}

	if !h.pool.Submit(evalReq) {
		slog.Warn("eval event dropped", "traceId", req.TraceID)
	}

	w.WriteHeader(http.StatusOK)
}

// handleMetrics returns aggregated quality metrics for the dashboard.
func (h *Handler) handleMetrics(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	collection := q.Get("collection")
	platform := q.Get("platform")
	periodDays := 0
	if p := q.Get("period"); p != "" {
		// Accept "7d", "30d", or plain int
		s := p
		if len(s) > 0 && s[len(s)-1] == 'd' {
			s = s[:len(s)-1]
		}
		periodDays, _ = strconv.Atoi(s)
	}

	result, err := h.store.QueryMetrics(r.Context(), store.MetricsQuery{
		Collection: collection,
		Platform:   platform,
		PeriodDays: periodDays,
	})
	if err != nil {
		slog.Error("metrics query failed", "error", err)
		http.Error(w, `{"error":"internal error"}`, http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	json.NewEncoder(w).Encode(result)
}

func (h *Handler) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status": "ok",
		"time":   time.Now().UTC().Format(time.RFC3339),
	})
}

// handleReset deletes all eval records (optionally scoped to ?collection=).
// Called by the dashboard Reset button.
func (h *Handler) handleReset(w http.ResponseWriter, r *http.Request) {
	collection := r.URL.Query().Get("collection")
	if err := h.store.Reset(r.Context(), collection); err != nil {
		slog.Error("reset failed", "error", err)
		http.Error(w, `{"error":"reset failed"}`, http.StatusInternalServerError)
		return
	}
	scope := "all collections"
	if collection != "" {
		scope = collection
	}
	slog.Info("eval store reset", "scope", scope)
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	json.NewEncoder(w).Encode(map[string]string{"status": "reset", "scope": scope})
}

// handleResults returns per-query scored rows for the dashboard table.
func (h *Handler) handleResults(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	collection := q.Get("collection")
	platform := q.Get("platform")
	limit := 500
	if l := q.Get("limit"); l != "" {
		if v, err := strconv.Atoi(l); err == nil && v > 0 {
			limit = v
		}
	}
	rows, err := h.store.QueryResults(r.Context(), store.ResultsQuery{
		Collection: collection,
		Platform:   platform,
		Limit:      limit,
		OrderDesc:  true,
	})
	if err != nil {
		slog.Error("results query failed", "error", err)
		http.Error(w, `{"error":"internal error"}`, http.StatusInternalServerError)
		return
	}
	if rows == nil {
		rows = []store.ResultRow{}
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	json.NewEncoder(w).Encode(rows)
}

// handlePlatforms returns distinct platform values (e.g. "flogo", "bw6").
func (h *Handler) handlePlatforms(w http.ResponseWriter, r *http.Request) {
	plats, err := h.store.ListPlatforms(r.Context())
	if err != nil {
		slog.Error("platforms query failed", "error", err)
		http.Error(w, `{"error":"internal error"}`, http.StatusInternalServerError)
		return
	}
	if plats == nil {
		plats = []string{}
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	json.NewEncoder(w).Encode(plats)
}

// handleCollections returns the distinct collection names.
func (h *Handler) handleCollections(w http.ResponseWriter, r *http.Request) {
	cols, err := h.store.ListCollections(r.Context())
	if err != nil {
		slog.Error("collections query failed", "error", err)
		http.Error(w, `{"error":"internal error"}`, http.StatusInternalServerError)
		return
	}
	if cols == nil {
		cols = []string{}
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	json.NewEncoder(w).Encode(cols)
}
