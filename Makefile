# Local hand-build for the test/dev phase. Reuses the EXACT same Dockerfile the
# CI docker-build job uses (bases/<BASE>/Dockerfile) — only the registry/tag
# differ. CI publishes beclab/audio-<BASE>; this pushes to your personal registry.
#
#   make build-push BASE=qwen TAG=dev1              # -> lovehunter9/audio-qwen:dev1
#   make build-push BASE=qwen TAG=dev1 REGISTRY=... # override registry
REGISTRY ?= docker.io/lovehunter9
BASE ?= qwen
TAG ?= dev
# Local-only escape hatch, e.g. behind a proxy:
#   EXTRA="--build-arg HTTP_PROXY=http://... --build-arg HTTPS_PROXY=http://..."
EXTRA ?=
COMMIT := $(shell git rev-parse --short HEAD 2>/dev/null || echo dev)
BUILD_DATE := $(shell date -u +%Y-%m-%dT%H:%M:%SZ)
IMAGE := $(REGISTRY)/audio-$(BASE):$(TAG)

.PHONY: build-push build lint
build-push:
	docker buildx build --platform linux/amd64 \
	  -f bases/$(BASE)/Dockerfile \
	  -t $(IMAGE) \
	  --build-arg VERSION=$(TAG) \
	  --build-arg COMMIT=$(COMMIT) \
	  --build-arg BUILD_DATE=$(BUILD_DATE) \
	  $(EXTRA) --push .

build:
	docker build \
	  -f bases/$(BASE)/Dockerfile \
	  -t $(IMAGE) \
	  --build-arg VERSION=$(TAG) \
	  --build-arg COMMIT=$(COMMIT) \
	  --build-arg BUILD_DATE=$(BUILD_DATE) $(EXTRA) .

lint:
	python -m compileall -q wrapper
