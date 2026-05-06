package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/mpandav-tibco/rageval/internal/engine"
	_ "modernc.org/sqlite"
)

const (
	// alertScoreThreshold is the overall score below which an eval is surfaced as a recent alert.
	alertScoreThreshold = 0.6
	// alertRecordLimit caps the number of recent alert rows returned per metrics query.
	alertRecordLimit = 10
)

// SQLiteStore persists eval results to a local SQLite database.
type SQLiteStore struct {
	db *sql.DB
}

func NewSQLiteStore(path string) (*SQLiteStore, error) {
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, fmt.Errorf("sqlite: open %s: %w", path, err)
	}
	db.SetMaxOpenConns(1) // SQLite is single-writer

	if err := migrate(db); err != nil {
		return nil, fmt.Errorf("sqlite: migrate: %w", err)
	}
	return &SQLiteStore{db: db}, nil
}

func migrate(db *sql.DB) error {
	// Create table if not exists (initial schema)
	_, err := db.Exec(`
CREATE TABLE IF NOT EXISTS eval_results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at          DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    pipeline_id         TEXT,
    trace_id            TEXT,
    collection          TEXT,
    query               TEXT,
    context_relevance   REAL,
    faithfulness        REAL,
    answer_relevance    REAL,
    chunk_utilization   REAL,
    answer_correctness  REAL,
    overall_score       REAL,
    flags               TEXT   -- JSON array
);
CREATE INDEX IF NOT EXISTS idx_collection ON eval_results(collection);
CREATE INDEX IF NOT EXISTS idx_created_at ON eval_results(created_at);
`)
	if err != nil {
		return err
	}

	// Additive migration: add LLM-judge columns and platform column if absent.
	// SQLite silently ignores ADD COLUMN if the column already exists is not supported;
	// we use a per-column table-info check instead.
	llmCols := []struct{ name, def string }{
		{"llm_ctx_relevance", "REAL"},
		{"llm_faithfulness", "REAL"},
		{"llm_ans_relevance", "REAL"},
		{"llm_overall", "REAL"},
		{"llm_reasoning", "TEXT"},
		{"platform", "TEXT"},
	}
	existing := map[string]bool{}
	rows, err := db.Query(`PRAGMA table_info(eval_results)`)
	if err != nil {
		return fmt.Errorf("migrate: table_info: %w", err)
	}
	for rows.Next() {
		var cid int
		var name, colType string
		var notNull int
		var dfltValue sql.NullString
		var pk int
		_ = rows.Scan(&cid, &name, &colType, &notNull, &dfltValue, &pk)
		existing[name] = true
	}
	rows.Close()

	for _, col := range llmCols {
		if !existing[col.name] {
			if _, err := db.Exec(fmt.Sprintf(`ALTER TABLE eval_results ADD COLUMN %s %s`, col.name, col.def)); err != nil {
				return fmt.Errorf("migrate: add column %s: %w", col.name, err)
			}
		}
	}
	return nil
}

func (s *SQLiteStore) WriteResult(ctx context.Context, r *engine.EvalResult) error {
	flags, _ := json.Marshal(r.Flags)
	now := time.Now().UTC().Format("2006-01-02T15:04:05.000")
	_, err := s.db.ExecContext(ctx, `
INSERT INTO eval_results
  (created_at, pipeline_id, platform, trace_id, collection, query,
   context_relevance, faithfulness, answer_relevance,
   chunk_utilization, answer_correctness, overall_score, flags,
   llm_ctx_relevance, llm_faithfulness, llm_ans_relevance, llm_overall, llm_reasoning)
VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`,
		now,
		r.PipelineID, r.Platform, r.TraceID, r.Collection, r.Query,
		r.ContextRelevance, r.Faithfulness, r.AnswerRelevance,
		r.ChunkUtilization, r.AnswerCorrectness, r.OverallScore,
		string(flags),
		r.LLMContextRelevance, r.LLMFaithfulness, r.LLMAnswerRelevance,
		r.LLMOverall, r.LLMReasoning,
	)
	return err
}

func (s *SQLiteStore) QueryMetrics(ctx context.Context, q MetricsQuery) (*MetricsResult, error) {
	if q.Limit == 0 {
		q.Limit = 1000
	}

	where := []string{"1=1"}
	args := []interface{}{}

	if q.Collection != "" {
		where = append(where, "collection = ?")
		args = append(args, q.Collection)
	}
	if q.Platform != "" {
		where = append(where, "platform = ?")
		args = append(args, q.Platform)
	}
	if q.PeriodDays > 0 {
		cutoff := time.Now().AddDate(0, 0, -q.PeriodDays).Format("2006-01-02")
		where = append(where, "created_at >= ?")
		args = append(args, cutoff)
	}

	clause := strings.Join(where, " AND ")

	row := s.db.QueryRowContext(ctx, fmt.Sprintf(`
SELECT
  COUNT(*),
  COALESCE(AVG(overall_score), 0),
  COALESCE(AVG(context_relevance), 0),
  COALESCE(AVG(faithfulness), 0),
  COALESCE(AVG(answer_relevance), 0),
  COALESCE(SUM(CASE WHEN flags != '[]' AND flags != 'null' THEN 1 ELSE 0 END), 0),
  COALESCE(AVG(llm_overall), 0),
  COALESCE(AVG(llm_ctx_relevance), 0),
  COALESCE(AVG(llm_faithfulness), 0),
  COALESCE(AVG(llm_ans_relevance), 0)
FROM eval_results WHERE %s`, clause), args...)

	var total, hallucCount int
	var avgOverall, avgCtx, avgFaith, avgAns float64
	var avgLLMOverall, avgLLMCtx, avgLLMFaith, avgLLMAns float64
	if err := row.Scan(&total, &avgOverall, &avgCtx, &avgFaith, &avgAns, &hallucCount,
		&avgLLMOverall, &avgLLMCtx, &avgLLMFaith, &avgLLMAns); err != nil {
		return nil, fmt.Errorf("query metrics: %w", err)
	}

	hallucinationPct := 0.0
	if total > 0 {
		hallucinationPct = float64(hallucCount) / float64(total) * 100
	}

	// Recent low-score alerts (overall < 0.6)
	alertArgs := append(args, alertScoreThreshold, alertRecordLimit)
	rows, err := s.db.QueryContext(ctx, fmt.Sprintf(`
SELECT trace_id, query, overall_score, flags
FROM eval_results
WHERE %s AND overall_score < ?
ORDER BY created_at DESC LIMIT ?`, clause), alertArgs...)
	if err != nil {
		return nil, fmt.Errorf("query alerts: %w", err)
	}
	defer rows.Close()

	var alerts []AlertRecord
	for rows.Next() {
		var a AlertRecord
		var flagsJSON string
		if err := rows.Scan(&a.TraceID, &a.Query, &a.OverallScore, &flagsJSON); err != nil {
			continue
		}
		_ = json.Unmarshal([]byte(flagsJSON), &a.Flags)
		alerts = append(alerts, a)
	}

	return &MetricsResult{
		Collection:         q.Collection,
		PeriodDays:         q.PeriodDays,
		TotalEvals:         total,
		AvgOverall:         round2(avgOverall),
		AvgContextRel:      round2(avgCtx),
		AvgFaithfulness:    round2(avgFaith),
		AvgAnswerRel:       round2(avgAns),
		HallucinationPct:   round2(hallucinationPct),
		AvgLLMOverall:      round2(avgLLMOverall),
		AvgLLMContextRel:   round2(avgLLMCtx),
		AvgLLMFaithfulness: round2(avgLLMFaith),
		AvgLLMAnswerRel:    round2(avgLLMAns),
		RecentAlerts:       alerts,
	}, nil
}

func (s *SQLiteStore) QueryResults(ctx context.Context, q ResultsQuery) ([]ResultRow, error) {
	if q.Limit == 0 {
		q.Limit = 500
	}
	where := []string{"1=1"}
	args := []interface{}{}
	if q.Collection != "" {
		where = append(where, "collection = ?")
		args = append(args, q.Collection)
	}
	if q.Platform != "" {
		where = append(where, "platform = ?")
		args = append(args, q.Platform)
	}
	clause := strings.Join(where, " AND ")
	order := "ASC"
	if q.OrderDesc {
		order = "DESC"
	}
	args = append(args, q.Limit)
	rows, err := s.db.QueryContext(ctx, fmt.Sprintf(`
SELECT id, created_at, COALESCE(trace_id,''), COALESCE(collection,''), COALESCE(query,''),
       COALESCE(overall_score,0), COALESCE(context_relevance,0),
       COALESCE(faithfulness,0), COALESCE(answer_relevance,0),
       COALESCE(llm_overall,0), COALESCE(llm_faithfulness,0),
       COALESCE(llm_ctx_relevance,0), COALESCE(llm_reasoning,''), COALESCE(flags,'[]')
FROM eval_results WHERE %s ORDER BY created_at %s LIMIT ?`, clause, order), args...)
	if err != nil {
		return nil, fmt.Errorf("query results: %w", err)
	}
	defer rows.Close()
	var result []ResultRow
	for rows.Next() {
		var r ResultRow
		var flagsJSON string
		if err := rows.Scan(
			&r.ID, &r.CreatedAt, &r.TraceID, &r.Collection, &r.Query,
			&r.OverallScore, &r.ContextRelevance, &r.Faithfulness, &r.AnswerRelevance,
			&r.LLMOverall, &r.LLMFaithfulness, &r.LLMContextRel, &r.LLMReasoning, &flagsJSON,
		); err != nil {
			continue
		}
		_ = json.Unmarshal([]byte(flagsJSON), &r.Flags)
		r.HasFlags = len(r.Flags) > 0
		result = append(result, r)
	}
	return result, nil
}

func (s *SQLiteStore) ListCollections(ctx context.Context) ([]string, error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT DISTINCT collection FROM eval_results WHERE collection != '' ORDER BY collection`)
	if err != nil {
		return nil, fmt.Errorf("list collections: %w", err)
	}
	defer rows.Close()
	var cols []string
	for rows.Next() {
		var c string
		_ = rows.Scan(&c)
		cols = append(cols, c)
	}
	return cols, nil
}

func (s *SQLiteStore) ListPlatforms(ctx context.Context) ([]string, error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT DISTINCT platform FROM eval_results WHERE platform != '' ORDER BY platform`)
	if err != nil {
		return nil, fmt.Errorf("list platforms: %w", err)
	}
	defer rows.Close()
	var plats []string
	for rows.Next() {
		var p string
		_ = rows.Scan(&p)
		plats = append(plats, p)
	}
	return plats, nil
}

func (s *SQLiteStore) Close() error {
	return s.db.Close()
}

// Reset deletes all eval records. Pass collection="" to wipe all collections.
func (s *SQLiteStore) Reset(ctx context.Context, collection string) error {
	var err error
	if collection == "" {
		_, err = s.db.ExecContext(ctx, `DELETE FROM eval_results`)
	} else {
		_, err = s.db.ExecContext(ctx, `DELETE FROM eval_results WHERE collection = ?`, collection)
	}
	return err
}

func round2(v float64) float64 {
	return float64(int(v*100+0.5)) / 100
}
