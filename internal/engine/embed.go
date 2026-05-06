package engine

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"time"
)

// EmbedProvider is a generic interface for any embedding backend.
type EmbedProvider interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
}

// OllamaProvider calls the Ollama embeddings API.
type OllamaProvider struct {
	BaseURL string
	Model   string
	client  *http.Client
}

func NewOllamaProvider(baseURL, model string, timeoutSec int) *OllamaProvider {
	return &OllamaProvider{
		BaseURL: baseURL,
		Model:   model,
		client:  &http.Client{Timeout: time.Duration(timeoutSec) * time.Second},
	}
}

type ollamaEmbedRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type ollamaEmbedResponse struct {
	Embeddings [][]float32 `json:"embeddings"`
}

// maxEmbedChars is the max characters sent per text to Ollama.
// nomic-embed-text has a 2048 token context; ~800 chars is a safe limit.
const maxEmbedChars = 800

// Embed calls the Ollama /api/embed endpoint with all texts in a single batch.
func (p *OllamaProvider) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	start := time.Now()

	truncated := make([]string, len(texts))
	for i, t := range texts {
		truncated[i] = truncate(t, maxEmbedChars)
	}

	body, _ := json.Marshal(ollamaEmbedRequest{Model: p.Model, Input: truncated})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		p.BaseURL+"/api/embed", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("ollama embed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama embed: status %d", resp.StatusCode)
	}

	var out ollamaEmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("ollama embed: decode: %w", err)
	}
	if len(out.Embeddings) != len(texts) {
		return nil, fmt.Errorf("ollama embed: expected %d embeddings, got %d", len(texts), len(out.Embeddings))
	}

	dim := 0
	if len(out.Embeddings) > 0 {
		dim = len(out.Embeddings[0])
	}
	slog.Debug("embed batch complete",
		"texts", len(texts),
		"dim", dim,
		"latency", time.Since(start).Round(time.Millisecond))

	return out.Embeddings, nil
}
