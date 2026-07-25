# Local hand-build for the test/dev phase. Reuses the EXACT same Dockerfile the
# CI docker-build job uses (bases/<BASE>/Dockerfile) — only the registry/tag
# differ. CI publishes beclab/audio-<BASE>; this pushes wherever you point it.
#
#   make build-push REGISTRY=docker.io/<ns> BASE=qwen TAG=dev1
#
# A base whose upstream image is too big to unpack on a build host has an
# append.env instead of a Dockerfile; build-push then publishes it through the
# registry with crane, exactly as CI does. See scripts/append-image.sh.
#
# REGISTRY is deliberately unset: dev images belong in your own namespace, so
# name it per invocation or export it from your shell.
REGISTRY ?=
BASE ?= qwen
TAG ?= dev
# Local-only escape hatch, e.g. behind a proxy:
#   EXTRA="--build-arg HTTP_PROXY=http://... --build-arg HTTPS_PROXY=http://..."
EXTRA ?=
COMMIT := $(shell git rev-parse --short HEAD 2>/dev/null || echo dev)
BUILD_DATE := $(shell date -u +%Y-%m-%dT%H:%M:%SZ)
IMAGE := $(REGISTRY)/audio-$(BASE):$(TAG)

.PHONY: build-push build lint require-registry
require-registry:
	@test -n "$(REGISTRY)" || { echo "REGISTRY is required, e.g. REGISTRY=docker.io/<ns>"; exit 1; }

build-push: require-registry
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
