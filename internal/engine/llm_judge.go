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
// Rubric scores (1–5) are normalized to 0–1 via (score-1)/4.
type LLMJudgeResult struct {
	ContextRelevance  float64
	Faithfulness      float64
	AnswerRelevance   float64
	ClaimFaithfulness float64 // grounded_claims / total_claims (0 if no claims returned)
	ContextRecall     float64 // covered_reference_facts / total_facts (0 if no expected answer)
	Overall           float64 // weighted: ctx*0.40 + faith*0.35 + ans*0.25
	Reasoning         string
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
		"claimFaithfulness", result.ClaimFaithfulness,
		"answerRelevance", result.AnswerRelevance,
		"contextRecall", result.ContextRecall,
		"overall", result.Overall)
	return result, nil
}

// buildJudgePrompt constructs the evaluation prompt for the LLM judge.
// Uses 1–5 rubric scales for reproducible scoring and requests claim-level
// faithfulness analysis and optional context recall in a single inference call.
func buildJudgePrompt(req EvalRequest) string {
	var chunksBuilder strings.Builder
	for i, c := range req.Chunks {
		chunksBuilder.WriteString(fmt.Sprintf("[Chunk %d] %s\n", i+1, truncate(c.Text, 400)))
	}

	refSection, refTask, refJSONField := "", "", ""
	if req.ExpectedAnswer != "" {
		refSection = "\nREFERENCE ANSWER (ideal response, for recall only):\n" +
			truncate(req.ExpectedAnswer, 400) + "\n"
		refTask = "\n\nTask 3 \u2014 Context Recall:\n" +
			"Decompose the REFERENCE ANSWER into distinct atomic facts. " +
			"For each fact, state whether the retrieved context chunks contain enough information to derive it."
		refJSONField = ",\n  \"reference_facts\": [{\"fact\": \"...\", \"covered\": true|false}, ...]"
	}

	return "You are a strict RAG quality evaluator.\n" +
		"Evaluate ONLY using the retrieved context chunks. Do NOT use your own knowledge.\n\n" +
		"QUESTION: " + req.Query + "\n\n" +
		"RETRIEVED CONTEXT CHUNKS:\n" + chunksBuilder.String() + "\n" +
		"GENERATED ANSWER:\n" + truncate(req.Answer, 500) + "\n" +
		refSection +
		"\n---\n" +
		"Task 1 \u2014 Rubric Scoring (score each dimension 1\u20135):\n\n" +
		"context_relevance \u2014 Do the retrieved chunks directly answer the question?\n" +
		"  5: All chunks highly relevant and specific  4: Most relevant, one or two tangential\n" +
		"  3: About half relevant  2: Only one or two chunks touch the question  1: Chunks unrelated\n\n" +
		"faithfulness \u2014 Are ALL claims in the answer explicitly supported by the chunks?\n" +
		"  5: Every claim traceable to a chunk  4: Nearly all grounded, one minor unsupported detail\n" +
		"  3: Most grounded, some unsupported content  2: Several key claims absent  1: Ignores chunks\n\n" +
		"answer_relevance \u2014 Does the answer directly and completely address the question?\n" +
		"  5: Complete, direct answer  4: Good with minor gaps  3: Partial  2: Tangential  1: Off-topic\n\n" +
		"Task 2 \u2014 Claim Faithfulness:\n" +
		"Decompose the GENERATED ANSWER into atomic factual claims. " +
		"For each claim, state whether it is explicitly supported by the retrieved context chunks." +
		refTask +
		"\n---\n" +
		"Respond ONLY with valid JSON (no prose, no markdown fences):\n" +
		"{\n  \"context_relevance\": <1-5>,\n  \"faithfulness\": <1-5>,\n  \"answer_relevance\": <1-5>,\n" +
		"  \"reasoning\": \"<one sentence explaining the faithfulness score>\",\n" +
		"  \"answer_claims\": [{\"claim\": \"...\", \"grounded\": true|false}, ...]" +
		refJSONField + "\n}"
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
		AnswerClaims     []struct {
			Claim    string `json:"claim"`
			Grounded bool   `json:"grounded"`
		} `json:"answer_claims"`
		ReferenceFacts []struct {
			Fact    string `json:"fact"`
			Covered bool   `json:"covered"`
		} `json:"reference_facts"`
	}
	if err := json.Unmarshal([]byte(raw), &parsed); err != nil {
		return nil, fmt.Errorf("parse JSON: %w", err)
	}

	// Normalize 1-5 rubric → 0-1 via (score-1)/4.
	// Guard against models that return 0-10 instead of 1-5.
	cr := clamp01((parsed.ContextRelevance - 1) / 4.0)
	fa := clamp01((parsed.Faithfulness - 1) / 4.0)
	ar := clamp01((parsed.AnswerRelevance - 1) / 4.0)
	if parsed.ContextRelevance > 5 {
		cr = clamp01(parsed.ContextRelevance / 10.0)
	}
	if parsed.Faithfulness > 5 {
		fa = clamp01(parsed.Faithfulness / 10.0)
	}
	if parsed.AnswerRelevance > 5 {
		ar = clamp01(parsed.AnswerRelevance / 10.0)
	}
	overall := cr*0.40 + fa*0.35 + ar*0.25

	// Claim-level faithfulness
	claimFaith := 0.0
	if len(parsed.AnswerClaims) > 0 {
		grounded := 0
		for _, c := range parsed.AnswerClaims {
			if c.Grounded {
				grounded++
			}
		}
		claimFaith = float64(grounded) / float64(len(parsed.AnswerClaims))
	}

	// Context recall (only populated when reference answer was provided)
	ctxRecall := 0.0
	if len(parsed.ReferenceFacts) > 0 {
		covered := 0
		for _, f := range parsed.ReferenceFacts {
			if f.Covered {
				covered++
			}
		}
		ctxRecall = float64(covered) / float64(len(parsed.ReferenceFacts))
	}

	return &LLMJudgeResult{
		ContextRelevance:  round2(cr),
		Faithfulness:      round2(fa),
		AnswerRelevance:   round2(ar),
		ClaimFaithfulness: round2(claimFaith),
		ContextRecall:     round2(ctxRecall),
		Overall:           round2(overall),
		Reasoning:         parsed.Reasoning,
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
