package main

import (
	"context"
	"embed"
	"fmt"
	"io/fs"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/mpandav-tibco/rageval/internal/api"
	"github.com/mpandav-tibco/rageval/internal/config"
	"github.com/mpandav-tibco/rageval/internal/engine"
	"github.com/mpandav-tibco/rageval/internal/queue"
	"github.com/mpandav-tibco/rageval/internal/store"
)

//go:embed dashboard
var dashboardFS embed.FS

func main() {
	// Config — path from env or default
	cfgPath := os.Getenv("RAGEVAL_CONFIG")
	if cfgPath == "" {
		cfgPath = "config.yaml"
	}
	cfg, err := config.Load(cfgPath)
	if err != nil {
		slog.Error("failed to load config", "path", cfgPath, "error", err)
		os.Exit(1)
	}

	// Log level: config → env override
	logLevel := slog.LevelInfo
	levelStr := cfg.Server.LogLevel
	if env := os.Getenv("RAGEVAL_LOG_LEVEL"); env != "" {
		levelStr = env
	}
	if strings.EqualFold(levelStr, "debug") {
		logLevel = slog.LevelDebug
	}
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: logLevel})))

	slog.Info("rageval starting",
		"port", cfg.Server.Port,
		"embedProvider", cfg.Embed.Provider,
		"embedModel", cfg.Embed.Model,
		"storage", cfg.Storage.Type,
		"workers", cfg.Eval.WorkerCount,
		"sampleRate", cfg.Eval.SampleRate)

	// Storage
	var st store.EvalStore
	switch cfg.Storage.Type {
	case "sqlite":
		st, err = store.NewSQLiteStore(cfg.Storage.Path)
		if err != nil {
			slog.Error("failed to open SQLite store", "path", cfg.Storage.Path, "error", err)
			os.Exit(1)
		}
	default:
		slog.Error("unsupported storage type", "type", cfg.Storage.Type)
		os.Exit(1)
	}
	defer st.Close()
	slog.Info("storage ready", "type", cfg.Storage.Type)

	// Embed provider
	var embedProvider engine.EmbedProvider
	switch cfg.Embed.Provider {
	case "ollama":
		embedProvider = engine.NewOllamaProvider(cfg.Embed.URL, cfg.Embed.Model, cfg.Embed.TimeoutSeconds)
	default:
		slog.Error("unsupported embed provider", "provider", cfg.Embed.Provider)
		os.Exit(1)
	}

	// Scorer + worker pool
	scorer := engine.NewScorer(embedProvider, cfg.Eval.FaithfulnessThreshold)
	if cfg.Eval.LLMJudge.Enabled {
		judge := engine.NewLLMJudge(cfg.Eval.LLMJudge.URL, cfg.Eval.LLMJudge.Model, cfg.Eval.LLMJudge.TimeoutSeconds)
		scorer.SetLLMJudge(judge)
		slog.Info("LLM judge enabled", "model", cfg.Eval.LLMJudge.Model, "url", cfg.Eval.LLMJudge.URL)
	}
	pool := queue.NewWorkerPool(
		cfg.Eval.ChannelBuffer,
		cfg.Eval.WorkerCount,
		scorer, st,
		cfg.Eval.SampleRate,
	)

	// HTTP server
	mux := http.NewServeMux()
	handler := api.NewHandler(pool, st)
	handler.RegisterRoutes(mux)

	// Serve dashboard at /
	dashSub, _ := fs.Sub(dashboardFS, "dashboard")
	mux.Handle("/", http.FileServer(http.FS(dashSub)))

	srv := &http.Server{
		Addr:         fmt.Sprintf(":%d", cfg.Server.Port),
		Handler:      mux,
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 10 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	// Graceful shutdown
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)

	go func() {
		slog.Info("HTTP server listening", "addr", srv.Addr)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			slog.Error("server error", "error", err)
			os.Exit(1)
		}
	}()

	<-stop
	slog.Info("shutting down...")

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	_ = srv.Shutdown(ctx)
	pool.Shutdown(10 * time.Second)

	slog.Info("rageval stopped")
}
