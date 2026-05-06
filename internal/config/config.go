package config

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

type Config struct {
	Server  ServerConfig  `yaml:"server"`
	Embed   EmbedConfig   `yaml:"embed"`
	Storage StorageConfig `yaml:"storage"`
	Eval    EvalConfig    `yaml:"eval"`
}

type ServerConfig struct {
	Port     int    `yaml:"port"`
	LogLevel string `yaml:"logLevel"` // info | debug (default: info)
}

type EmbedConfig struct {
	Provider       string `yaml:"provider"`
	URL            string `yaml:"url"`
	Model          string `yaml:"model"`
	BatchSize      int    `yaml:"batchSize"`
	TimeoutSeconds int    `yaml:"timeoutSeconds"`
}

type StorageConfig struct {
	Type string `yaml:"type"` // sqlite | postgres
	Path string `yaml:"path"` // for sqlite
	DSN  string `yaml:"dsn"`  // for postgres
}

type EvalConfig struct {
	WorkerCount           int            `yaml:"workerCount"`
	ChannelBuffer         int            `yaml:"channelBuffer"`
	SampleRate            float64        `yaml:"sampleRate"`
	FaithfulnessThreshold float64        `yaml:"faithfulnessThreshold"`
	LLMJudge              LLMJudgeConfig `yaml:"llmJudge"`
}

// LLMJudgeConfig controls the optional LLM-as-a-judge scoring layer.
type LLMJudgeConfig struct {
	Enabled        bool   `yaml:"enabled"`
	URL            string `yaml:"url"`
	Model          string `yaml:"model"`
	TimeoutSeconds int    `yaml:"timeoutSeconds"`
}

func Load(path string) (*Config, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("config: open %s: %w", path, err)
	}
	defer f.Close()

	cfg := &Config{}
	if err := yaml.NewDecoder(f).Decode(cfg); err != nil {
		return nil, fmt.Errorf("config: decode: %w", err)
	}
	applyDefaults(cfg)
	return cfg, nil
}

func applyDefaults(c *Config) {
	if c.Server.Port == 0 {
		c.Server.Port = 8080
	}
	if c.Embed.Provider == "" {
		c.Embed.Provider = "ollama"
	}
	if c.Embed.URL == "" {
		c.Embed.URL = "http://localhost:11434"
	}
	if c.Embed.Model == "" {
		c.Embed.Model = "nomic-embed-text"
	}
	if c.Embed.BatchSize == 0 {
		c.Embed.BatchSize = 32
	}
	if c.Embed.TimeoutSeconds == 0 {
		c.Embed.TimeoutSeconds = 30
	}
	if c.Storage.Type == "" {
		c.Storage.Type = "sqlite"
	}
	if c.Eval.LLMJudge.URL == "" {
		c.Eval.LLMJudge.URL = "http://localhost:11434"
	}
	if c.Eval.LLMJudge.Model == "" {
		c.Eval.LLMJudge.Model = "llama3.1:8b"
	}
	if c.Eval.LLMJudge.TimeoutSeconds == 0 {
		c.Eval.LLMJudge.TimeoutSeconds = 60
	}
	if c.Storage.Type == "sqlite" && c.Storage.Path == "" {
		c.Storage.Path = "./rageval.db"
	}
	if c.Eval.WorkerCount == 0 {
		c.Eval.WorkerCount = 2
	}
	if c.Eval.ChannelBuffer == 0 {
		c.Eval.ChannelBuffer = 1000
	}
	if c.Eval.SampleRate == 0 {
		c.Eval.SampleRate = 1.0
	}
	if c.Eval.FaithfulnessThreshold == 0 {
		c.Eval.FaithfulnessThreshold = 0.75
	}
}
