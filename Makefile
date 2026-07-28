# make build-push REGISTRY=docker.io/<ns> BASE=qwen TAG=dev1 — CI's Dockerfile, your own namespace.
REGISTRY ?=
BASE ?= qwen
TAG ?= dev
# Local escape hatch, e.g. EXTRA="--build-arg HTTPS_PROXY=http://..." behind a proxy.
EXTRA ?=
COMMIT := $(shell git rev-parse --short HEAD 2>/dev/null || echo dev)
BUILD_DATE := $(shell date -u +%Y-%m-%dT%H:%M:%SZ)
IMAGE := $(REGISTRY)/audio-$(BASE):$(TAG)

.PHONY: build-push build deps lint require-registry
require-registry:
	@test -n "$(REGISTRY)" || { echo "REGISTRY is required, e.g. REGISTRY=docker.io/<ns>"; exit 1; }

# Publish the deps image if its recipe changed; the tag IS the recipe's hash, so this is a no-op
# until you edit deps.Dockerfile, and the slow build happens exactly once per edit.
deps: require-registry
	@test -f bases/$(BASE)/deps.Dockerfile || exit 0; \
	IMG=$$(./scripts/deps-image.sh $(BASE) $(REGISTRY)/audio-$(BASE)); \
	if crane manifest "$$IMG" >/dev/null 2>&1; then echo "deps up to date: $$IMG"; else \
	  echo "building deps $$IMG"; \
	  docker buildx build --platform linux/amd64 -f bases/$(BASE)/deps.Dockerfile \
	    -t "$$IMG" $(EXTRA) --push .; \
	fi

build-push: require-registry deps
	@if [ -f bases/$(BASE)/append.env ]; then \
	  BASE=$(BASE) IMAGE=$(IMAGE) VERSION=$(TAG) COMMIT=$(COMMIT) \
	    BUILD_DATE=$(BUILD_DATE) ./scripts/append-image.sh; \
	else \
	  docker buildx build --platform linux/amd64 \
	    -f bases/$(BASE)/Dockerfile \
	    -t $(IMAGE) \
	    --build-arg VERSION=$(TAG) \
	    --build-arg COMMIT=$(COMMIT) \
	    --build-arg BUILD_DATE=$(BUILD_DATE) \
	    $(EXTRA) --push .; \
	fi

build: require-registry
	docker build \
	  -f bases/$(BASE)/Dockerfile \
	  -t $(IMAGE) \
	  --build-arg VERSION=$(TAG) \
	  --build-arg COMMIT=$(COMMIT) \
	  --build-arg BUILD_DATE=$(BUILD_DATE) $(EXTRA) .

lint:
	python -m compileall -q wrapper
