# ── Etapa 1: Compilación ──────────────────────────────────────────────────────
FROM docker.io/library/rust:alpine3.23 AS builder

RUN apk add --no-cache --update \
            build-base \
            autoconf \
            gdb \
            musl-dev \
            pkgconfig \
            strace \
            openssl \
            openssl-dev \
            openssl-libs-static

WORKDIR /app

COPY Cargo.toml Cargo.lock ./
COPY src/ ./src/

RUN cargo build --release && \
    strip target/release/mcp-recipes

# ── Etapa 2: Runtime ──────────────────────────────────────────────────────────
FROM docker.io/library/alpine:3.23

RUN apk add --update --no-cache \
    ca-certificates \
    curl \
    && \
    adduser -S -u 1000 -D mcp

COPY --from=builder /app/target/release/mcp-recipes /usr/local/bin/

USER mcp
EXPOSE 3011

CMD ["mcp-recipes"]