# ── Build stage ──────────────────────────────────────────────────────────────
FROM golang:1.25-alpine AS builder
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags='-s -w' -o /rageval .

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM gcr.io/distroless/static-debian12
COPY --from=builder /rageval /rageval
COPY config.yaml /etc/rageval/config.yaml
ENTRYPOINT ["/rageval"]
