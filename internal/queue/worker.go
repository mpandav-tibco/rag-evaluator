package queue

import (
	"context"
	"log/slog"
	"math/rand"
	"sync"
	"time"

	"github.com/mpandav-tibco/rageval/internal/engine"
	"github.com/mpandav-tibco/rageval/internal/store"
)

// scoringTimeout is the maximum time a single eval job may take.
// Must be longer than llmJudge.timeoutSeconds (currently 180s) plus embed latency.
const scoringTimeout = 300 * time.Second

// EvalJob is a single evaluation job pushed from the API handler.
type EvalJob struct {
	ReceivedAt time.Time
	Request    engine.EvalRequest
}

// WorkerPool processes EvalJobs asynchronously off the hot path.
type WorkerPool struct {
	ch         chan EvalJob
	scorer     *engine.Scorer
	store      store.EvalStore
	sampleRate float64
	wg         sync.WaitGroup
}

func NewWorkerPool(bufSize, workers int, scorer *engine.Scorer, st store.EvalStore, sampleRate float64) *WorkerPool {
	p := &WorkerPool{
		ch:         make(chan EvalJob, bufSize),
		scorer:     scorer,
		store:      st,
		sampleRate: sampleRate,
	}
	for i := 0; i < workers; i++ {
		p.wg.Add(1)
		go p.run(i)
	}
	return p
}

// Submit enqueues a job. Non-blocking — drops if channel is full.
// Returns false if the job was dropped (channel full or sample rate filtered).
func (p *WorkerPool) Submit(req engine.EvalRequest) bool {
	if p.sampleRate < 1.0 && rand.Float64() > p.sampleRate {
		return false
	}
	select {
	case p.ch <- EvalJob{ReceivedAt: time.Now(), Request: req}:
		return true
	default:
		slog.Warn("rageval: eval queue full, dropping job", "traceId", req.TraceID)
		return false
	}
}

// Shutdown drains the channel and waits for all workers to finish.
func (p *WorkerPool) Shutdown(timeout time.Duration) {
	close(p.ch)
	done := make(chan struct{})
	go func() {
		p.wg.Wait()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(timeout):
		slog.Warn("rageval: worker pool shutdown timed out")
	}
}

func (p *WorkerPool) run(id int) {
	defer p.wg.Done()
	for job := range p.ch {
		p.process(id, job)
	}
}

func (p *WorkerPool) process(workerID int, job EvalJob) {
	waitTime := time.Since(job.ReceivedAt)
	slog.Debug("worker: processing job", "worker", workerID, "traceId", job.Request.TraceID, "queueWait", waitTime.Round(time.Millisecond))

	ctx, cancel := context.WithTimeout(context.Background(), scoringTimeout)
	defer cancel()

	result, err := p.scorer.Score(ctx, job.Request)
	if err != nil {
		slog.Error("rageval: scoring failed",
			"worker", workerID,
			"traceId", job.Request.TraceID,
			"error", err)
		return
	}

	latency := time.Since(job.ReceivedAt)
	slog.Info("rageval: eval complete",
		"traceId", result.TraceID,
		"collection", result.Collection,
		"overall", result.OverallScore,
		"faithfulness", result.Faithfulness,
		"contextRelevance", result.ContextRelevance,
		"latency", latency.Round(time.Millisecond))

	if err := p.store.WriteResult(context.Background(), result); err != nil {
		slog.Error("rageval: store write failed",
			"traceId", result.TraceID,
			"error", err)
	}
}
