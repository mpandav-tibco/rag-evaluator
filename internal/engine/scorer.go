package engine

import (
	"context"
	"fmt"
	"log/slog"
	"math"
	"strings"
)

// EvalRequest is the input to the scoring engine.
type EvalRequest struct {
	PipelineID     string
	Platform       string
	TraceID        string
	Collection     string
	Query          string
	Chunks         []Chunk
	Answer         string
	ExpectedAnswer string // optional — enables answerCorrectness metric
}

// Chunk is a single retrieved context chunk.
type Chunk struct {
	ID    string
	Text  string
	Score float64 // retrieval score from the vector DB
}

// EvalResult holds all computed quality metrics.
type EvalResult struct {
	PipelineID        string
	Platform          string
	TraceID           string
	Collection        string
	Query             string
	Answer            string
	ContextRelevance  float64 // mean cosine(query, chunk[i])
	Faithfulness      float64 // % of answer sentences grounded in chunks
	AnswerRelevance   float64 // cosine(query, answer)
	ChunkUtilization  float64 // cosine(answer, concat(chunks))
	AnswerCorrectness float64 // cosine(answer, expectedAnswer) — 0 if not provided
	ContextPrecision  float64 // Weighted Precision@K — penalizes poor retrieval ranking
	ContextRecall     float64 // fraction of expected-answer facts covered by context (needs ExpectedAnswer)
	OverallScore      float64
	Flags             []string // sentences flagged as not grounded

	// LLM-as-a-judge scores (populated when judge is enabled)
	LLMContextRelevance  float64
	LLMFaithfulness      float64
	LLMAnswerRelevance   float64
	LLMClaimFaithfulness float64 // claim-level: grounded claims / total claims
	LLMOverall           float64
	LLMReasoning         string
}

// Scorer computes RAG quality metrics using embedding-based methods only.
type Scorer struct {
	embed           EmbedProvider
	threshold       float64   // faithfulness threshold (default 0.75)
	aggregationMode string    // "mean" | "min" | "max" over per-chunk similarity scores
	judge           *LLMJudge // optional LLM-as-a-judge layer; nil = disabled
}

func NewScorer(embed EmbedProvider, faithfulnessThreshold float64, aggregationMode string) *Scorer {
	if aggregationMode == "" {
		aggregationMode = "mean"
	}
	return &Scorer{embed: embed, threshold: faithfulnessThreshold, aggregationMode: aggregationMode}
}

// SetLLMJudge attaches an LLM judge to the scorer. Call before first use.
func (s *Scorer) SetLLMJudge(j *LLMJudge) { s.judge = j }

// Score computes all quality metrics for a RAG event.
func (s *Scorer) Score(ctx context.Context, req EvalRequest) (*EvalResult, error) {
	if len(req.Chunks) == 0 {
		return nil, fmt.Errorf("scorer: no chunks provided")
	}

	// Collect all texts to embed in one batch where possible
	chunkTexts := make([]string, len(req.Chunks))
	for i, c := range req.Chunks {
		chunkTexts[i] = c.Text
	}
	concatChunks := strings.Join(chunkTexts, " ")

	// Build embed batch: query, answer, concat(chunks), each chunk
	batch := []string{req.Query, req.Answer, concatChunks}
	batch = append(batch, chunkTexts...)
	if req.ExpectedAnswer != "" {
		batch = append(batch, req.ExpectedAnswer)
	}

	vecs, err := s.embed.Embed(ctx, batch)
	if err != nil {
		return nil, fmt.Errorf("scorer: embed batch: %w", err)
	}

	queryVec := vecs[0]
	answerVec := vecs[1]
	concatVec := vecs[2]
	chunkVecs := vecs[3 : 3+len(req.Chunks)]

	// 1. Context Relevance — aggregated cosine(query, each chunk)
	chunkSims := make([]float64, len(chunkVecs))
	for i, cv := range chunkVecs {
		chunkSims[i] = cosineSim(queryVec, cv)
	}
	contextRel := aggregateChunkScores(chunkSims, s.aggregationMode)

	// 1b. Context Precision@K — weighted precision penalising low-ranked relevant chunks
	contextPrec := s.scoreContextPrecision(queryVec, chunkVecs)

	// 2. Answer Relevance — cosine(query, answer)
	answerRel := cosineSim(queryVec, answerVec)

	// 3. Chunk Utilization — cosine(answer, concat(chunks))
	chunkUtil := cosineSim(answerVec, concatVec)

	// 4. Faithfulness — per sentence: max cosine(sentence, chunk[]) >= threshold
	faithfulness, flags := s.scoreFaithfulness(ctx, req.Answer, chunkVecs)

	// 5. Answer Correctness (optional)
	answerCorrectness := 0.0
	if req.ExpectedAnswer != "" {
		expectedVec := vecs[3+len(req.Chunks)]
		answerCorrectness = cosineSim(answerVec, expectedVec)
	}

	// Overall: weighted average (correctness excluded if not provided)
	overall := (contextRel*0.25 + contextPrec*0.10 + faithfulness*0.35 + answerRel*0.20 + chunkUtil*0.10)
	if req.ExpectedAnswer != "" {
		// Blend in correctness at the expense of the others
		overall = (contextRel*0.15 + contextPrec*0.10 + faithfulness*0.25 + answerRel*0.15 +
			chunkUtil*0.05 + answerCorrectness*0.30)
	}

	result := &EvalResult{
		PipelineID:        req.PipelineID,
		Platform:          req.Platform,
		TraceID:           req.TraceID,
		Collection:        req.Collection,
		Query:             req.Query,
		Answer:            req.Answer,
		ContextRelevance:  round2(contextRel),
		ContextPrecision:  round2(contextPrec),
		Faithfulness:      round2(faithfulness),
		AnswerRelevance:   round2(answerRel),
		ChunkUtilization:  round2(chunkUtil),
		AnswerCorrectness: round2(answerCorrectness),
		OverallScore:      round2(overall),
		Flags:             flags,
	}

	slog.Debug("scorer: metrics computed",
		"traceId", req.TraceID,
		"contextRelevance", result.ContextRelevance,
		"contextPrecision", result.ContextPrecision,
		"faithfulness", result.Faithfulness,
		"answerRelevance", result.AnswerRelevance,
		"chunkUtilization", result.ChunkUtilization,
		"answerCorrectness", result.AnswerCorrectness,
		"overall", result.OverallScore,
		"flagged", len(result.Flags))

	// LLM-as-a-judge layer (optional, non-blocking on failure)
	if s.judge != nil {
		if llmResult, err := s.judge.Judge(ctx, req); err != nil {
			slog.Warn("scorer: llm judge failed", "traceId", req.TraceID, "error", err)
		} else {
			result.LLMContextRelevance = llmResult.ContextRelevance
			result.LLMFaithfulness = llmResult.Faithfulness
			result.LLMAnswerRelevance = llmResult.AnswerRelevance
			result.LLMClaimFaithfulness = llmResult.ClaimFaithfulness
			result.LLMOverall = llmResult.Overall
			result.LLMReasoning = llmResult.Reasoning
			result.ContextRecall = llmResult.ContextRecall
		}
	}

	return result, nil
}

// scoreFaithfulness checks each sentence in the answer against all chunk vectors.
// A sentence is "grounded" if its cosine similarity to any chunk >= threshold.
func (s *Scorer) scoreFaithfulness(ctx context.Context, answer string, chunkVecs [][]float32) (float64, []string) {
	sentences := splitSentences(answer)
	if len(sentences) == 0 {
		return 1.0, nil
	}

	vecs, err := s.embed.Embed(ctx, sentences)
	if err != nil {
		// If embed fails for faithfulness, return neutral score rather than fail entire eval
		return 0.5, []string{"faithfulness scoring unavailable: " + err.Error()}
	}

	grounded := 0
	var flags []string
	for i, sv := range vecs {
		maxSim := 0.0
		for _, cv := range chunkVecs {
			if sim := cosineSim(sv, cv); sim > maxSim {
				maxSim = sim
			}
		}
		slog.Debug("scorer: faithfulness sentence",
			"sentence", truncate(sentences[i], 60),
			"maxSim", fmt.Sprintf("%.3f", maxSim),
			"grounded", maxSim >= s.threshold)
		if maxSim >= s.threshold {
			grounded++
		} else {
			flags = append(flags, fmt.Sprintf("not grounded (sim=%.2f): %q", maxSim, truncate(sentences[i], 80)))
		}
	}
	return float64(grounded) / float64(len(sentences)), flags
}

// cosineSim computes cosine similarity between two float32 vectors.
func cosineSim(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0
	}
	return dot / denom
}

// aggregateChunkScores combines per-chunk similarity scores using the configured mode.
func aggregateChunkScores(scores []float64, mode string) float64 {
	if len(scores) == 0 {
		return 0
	}
	switch mode {
	case "min":
		m := scores[0]
		for _, s := range scores[1:] {
			if s < m {
				m = s
			}
		}
		return m
	case "max":
		m := scores[0]
		for _, s := range scores[1:] {
			if s > m {
				m = s
			}
		}
		return m
	default: // mean
		sum := 0.0
		for _, s := range scores {
			sum += s
		}
		return sum / float64(len(scores))
	}
}

// scoreContextPrecision computes Weighted Precision@K for the retrieved chunks.
// Chunks are assumed to be in retrieval-rank order (index 0 = rank 1).
// A chunk is "relevant" when cosine(query, chunk) >= s.threshold.
// This penalizes relevant chunks appearing late in the ranked list.
func (s *Scorer) scoreContextPrecision(queryVec []float32, chunkVecs [][]float32) float64 {
	K := len(chunkVecs)
	if K == 0 {
		return 0
	}
	relevant := make([]bool, K)
	totalRelevant := 0
	for i, cv := range chunkVecs {
		if cosineSim(queryVec, cv) >= s.threshold {
			relevant[i] = true
			totalRelevant++
		}
	}
	if totalRelevant == 0 {
		return 0
	}
	sum := 0.0
	relevantSoFar := 0
	for k := 0; k < K; k++ {
		if relevant[k] {
			relevantSoFar++
			sum += float64(relevantSoFar) / float64(k+1)
		}
	}
	return sum / float64(totalRelevant)
}

// splitSentences splits text on sentence-ending punctuation.
func splitSentences(text string) []string {
	var sentences []string
	for _, s := range strings.FieldsFunc(text, func(r rune) bool {
		return r == '.' || r == '!' || r == '?'
	}) {
		s = strings.TrimSpace(s)
		if len(s) > 10 { // skip very short fragments
			sentences = append(sentences, s)
		}
	}
	return sentences
}

func round2(v float64) float64 {
	return math.Round(v*100) / 100
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
