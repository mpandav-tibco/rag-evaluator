package store

import (
	"context"

	"github.com/mpandav-tibco/rageval/internal/engine"
)

// EvalStore is the persistence interface — swappable between SQLite, Postgres, etc.
type EvalStore interface {
	WriteResult(ctx context.Context, r *engine.EvalResult) error
	QueryMetrics(ctx context.Context, q MetricsQuery) (*MetricsResult, error)
	QueryResults(ctx context.Context, q ResultsQuery) ([]ResultRow, error)
	ListCollections(ctx context.Context) ([]string, error)
	// Reset deletes all evaluation records, optionally scoped to a collection.
	// Pass an empty string to wipe everything.
	Reset(ctx context.Context, collection string) error
	// ListPlatforms returns distinct platform values stored in eval_results.
	ListPlatforms(ctx context.Context) ([]string, error)
	// QueryCompare returns side-by-side statistics for two collections or platforms.
	QueryCompare(ctx context.Context, q CompareQuery) (*CompareResult, error)
	Close() error
}

// MetricsQuery defines filters for the dashboard metrics endpoint.
type MetricsQuery struct {
	Collection string
	Platform   string
	PeriodDays int // 0 = all time
	Limit      int // max records to scan (0 = 1000)
}

// MetricsResult is returned by QueryMetrics.
type MetricsResult struct {
	Collection          string  `json:"collection"`
	PeriodDays          int     `json:"periodDays"`
	TotalEvals          int     `json:"totalEvals"`
	AvgOverall          float64 `json:"avgOverall"`
	AvgContextRel       float64 `json:"avgContextRelevance"`
	AvgContextPrecision float64 `json:"avgContextPrecision"`
	AvgFaithfulness     float64 `json:"avgFaithfulness"`
	AvgAnswerRel        float64 `json:"avgAnswerRelevance"`
	AvgContextRecall    float64 `json:"avgContextRecall"`
	HallucinationPct    float64 `json:"hallucinationPct"` // % evals with flags
	// LLM-as-a-judge averages (0 when judge is disabled)
	AvgLLMOverall           float64       `json:"avgLLMOverall"`
	AvgLLMContextRel        float64       `json:"avgLLMContextRelevance"`
	AvgLLMFaithfulness      float64       `json:"avgLLMFaithfulness"`
	AvgLLMClaimFaithfulness float64       `json:"avgLLMClaimFaithfulness"`
	AvgLLMAnswerRel         float64       `json:"avgLLMAnswerRelevance"`
	RecentAlerts            []AlertRecord `json:"recentAlerts"`
}

// AlertRecord is a low-scoring eval surfaced on the dashboard.
type AlertRecord struct {
	TraceID      string   `json:"traceId"`
	Query        string   `json:"query"`
	OverallScore float64  `json:"overallScore"`
	Flags        []string `json:"flags"`
}

// ResultsQuery defines filters for per-query result rows.
type ResultsQuery struct {
	Collection string
	Platform   string
	Limit      int  // 0 → 500
	OrderDesc  bool // order by created_at DESC
}

// ResultRow is a single scored eval row returned by QueryResults.
type ResultRow struct {
	ID                   int      `json:"id"`
	CreatedAt            string   `json:"createdAt"`
	TraceID              string   `json:"traceId"`
	Collection           string   `json:"collection"`
	Query                string   `json:"query"`
	Answer               string   `json:"answer"`
	OverallScore         float64  `json:"overallScore"`
	ContextRelevance     float64  `json:"contextRelevance"`
	ContextPrecision     float64  `json:"contextPrecision"`
	ContextRecall        float64  `json:"contextRecall"`
	Faithfulness         float64  `json:"faithfulness"`
	AnswerRelevance      float64  `json:"answerRelevance"`
	LLMOverall           float64  `json:"llmOverall"`
	LLMFaithfulness      float64  `json:"llmFaithfulness"`
	LLMClaimFaithfulness float64  `json:"llmClaimFaithfulness"`
	LLMContextRel        float64  `json:"llmContextRelevance"`
	LLMAnswerRel         float64  `json:"llmAnswerRelevance"`
	LLMReasoning         string   `json:"llmReasoning"`
	HasFlags             bool     `json:"hasFlags"`
	Flags                []string `json:"flags"`
}

// CompareQuery defines two named groups to compare side-by-side.
type CompareQuery struct {
	ACollection string
	BCollection string
	APlatform   string
	BPlatform   string
	PeriodDays  int // 0 = all time
}

// RunStats contains aggregate statistics for one side of a comparison.
type RunStats struct {
	Name                    string  `json:"name"`
	TotalEvals              int     `json:"totalEvals"`
	AvgOverall              float64 `json:"avgOverall"`
	P25Overall              float64 `json:"p25Overall"`
	P75Overall              float64 `json:"p75Overall"`
	AvgContextRel           float64 `json:"avgContextRelevance"`
	AvgContextPrecision     float64 `json:"avgContextPrecision"`
	AvgFaithfulness         float64 `json:"avgFaithfulness"`
	AvgLLMClaimFaithfulness float64 `json:"avgLLMClaimFaithfulness"`
	AvgAnswerRel            float64 `json:"avgAnswerRelevance"`
	AvgLLMOverall           float64 `json:"avgLLMOverall"`
	HallucinationPct        float64 `json:"hallucinationPct"`
}

// CompareResult holds statistics for both sides of a collection/platform comparison.
type CompareResult struct {
	A RunStats `json:"a"`
	B RunStats `json:"b"`
}
