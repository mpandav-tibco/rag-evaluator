package engine

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"time"
)

// LLMJudgeResult holds scores produced by the LLM judge.
// Scores are normalized to 0-1 from the raw 0-10 LLM ratings.
type LLMJudgeResult struct {
	ContextRelevance float64
	Faithfulness     float64
	AnswerRelevance  float64
	Overall          float64 // weighted: ctx*0.40 + faith*0.35 + ans*0.25
	Reasoning        string
}

// LLMJudge evaluates RAG quality using a locally-hosted Ollama LLM.
// It uses a single structured prompt requesting JSON output so that one
// inference call covers all three evaluation dimensions.
type LLMJudge struct {
	BaseURL string
	Model   string
	client  *http.Client
}

// NewLLMJudge creates an LLMJudge backed by the given Ollama endpoint.
func NewLLMJudge(baseURL, model string, timeoutSec int) *LLMJudge {
	return &LLMJudge{
		BaseURL: strings.TrimRight(baseURL, "/"),
		Model:   model,
		client:  &http.Client{Timeout: time.Duration(timeoutSec) * time.Second},
	}
}

// Judge evaluates a RAG response and returns quality scores from the LLM.
func (j *LLMJudge) Judge(ctx context.Context, req EvalRequest) (*LLMJudgeResult, error) {
	prompt := buildJudgePrompt(req)
	slog.Debug("llm judge: sending prompt", "model", j.Model, "promptChars", len(prompt), "traceId", req.TraceID)

	body, _ := json.Marshal(map[string]interface{}{
		"model":  j.Model,
		"prompt": prompt,
		"stream": false,
		"format": "json",
	})

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		j.BaseURL+"/api/generate", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("llm_judge: build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := j.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("llm_judge: ollama call: %w", err)
	}
	defer resp.Body.Close()

	var ollamaResp struct {
		Response string `json:"response"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&ollamaResp); err != nil {
		return nil, fmt.Errorf("llm_judge: decode ollama response: %w", err)
	}

	result, err := parseJudgeResponse(ollamaResp.Response)
	if err != nil {
		slog.Warn("llm_judge: parse failed", "raw", truncate(ollamaResp.Response, 200), "error", err)
		return nil, err
	}
	slog.Debug("llm judge: scores",
		"traceId", req.TraceID,
		"contextRelevance", result.ContextRelevance,
		"faithfulness", result.Faithfulness,
		"answerRelevance", result.AnswerRelevance,
		"overall", result.Overall)
	return result, nil
}

// buildJudgePrompt constructs the evaluation prompt for the LLM judge.
// Chunks and answer are truncated to stay within practical context limits.
func buildJudgePrompt(req EvalRequest) string {
	var chunksBuilder strings.Builder
	for i, c := range req.Chunks {
		chunksBuilder.WriteString(fmt.Sprintf("[Chunk %d] %s\n", i+1, truncate(c.Text, 500)))
	}

	return fmt.Sprintf(
		`You are a strict RAG (Retrieval-Augmented Generation) quality evaluator.
Your task is to score the answer ONLY based on the retrieved context chunks provided below.
Do NOT use your own training knowledge — if a claim is correct but absent from the chunks, it still counts as ungrounded.

QUESTION: %s

RETRIEVED CONTEXT CHUNKS:
%s
ANSWER (generated from the retrieved context above):
%s

Score each dimension 0–10 based solely on the chunks above:
- context_relevance: Do the retrieved chunks contain information that specifically answers the question? (10 = direct answer in chunks; 0 = chunks are unrelated)
- faithfulness: Are ALL claims in the answer traceable to the retrieved chunks? Penalise any claim not found in the chunks, even if factually correct. (10 = every claim is in the chunks; 0 = answer ignores the chunks entirely)
- answer_relevance: Does the answer directly and completely address the question? (10 = complete direct answer; 0 = does not address it)

Respond ONLY with valid JSON, no other text:
{"context_relevance": <0-10>, "faithfulness": <0-10>, "answer_relevance": <0-10>, "reasoning": "<one concise sentence explaining the faithfulness score>"}`,
		req.Query,
		chunksBuilder.String(),
		truncate(req.Answer, 600),
	)
}

// parseJudgeResponse extracts and normalizes scores from the LLM JSON output.
func parseJudgeResponse(raw string) (*LLMJudgeResult, error) {
	// Strip markdown code fences some models emit despite format:json
	raw = strings.TrimSpace(raw)
	raw = strings.TrimPrefix(raw, "```json")
	raw = strings.TrimPrefix(raw, "```")
	raw = strings.TrimSuffix(raw, "```")
	raw = strings.TrimSpace(raw)

	// Extract first JSON object in case of extra text
	if i := strings.Index(raw, "{"); i > 0 {
		raw = raw[i:]
	}
	if i := strings.LastIndex(raw, "}"); i >= 0 && i < len(raw)-1 {
		raw = raw[:i+1]
	}

	var parsed struct {
		ContextRelevance float64 `json:"context_relevance"`
		Faithfulness     float64 `json:"faithfulness"`
		AnswerRelevance  float64 `json:"answer_relevance"`
		Reasoning        string  `json:"reasoning"`
	}
	if err := json.Unmarshal([]byte(raw), &parsed); err != nil {
		return nil, fmt.Errorf("parse JSON: %w", err)
	}

	// Normalize 0-10 → 0-1
	cr := clamp01(parsed.ContextRelevance / 10.0)
	fa := clamp01(parsed.Faithfulness / 10.0)
	ar := clamp01(parsed.AnswerRelevance / 10.0)
	overall := cr*0.40 + fa*0.35 + ar*0.25

	return &LLMJudgeResult{
		ContextRelevance: round2(cr),
		Faithfulness:     round2(fa),
		AnswerRelevance:  round2(ar),
		Overall:          round2(overall),
		Reasoning:        parsed.Reasoning,
	}, nil
}

func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}
