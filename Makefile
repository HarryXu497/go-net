GOLANGCI_LINT ?= golangci-lint
GOLANGCI_LINT_CONFIG ?= .golangci.yml

lint:
	@$(GOLANGCI_LINT) run --config $(GOLANGCI_LINT_CONFIG)

l: lint

lint-fix:
	@$(GOLANGCI_LINT) run --config $(GOLANGCI_LINT_CONFIG) --fix

lf: lint-fix

test:
	@go test -v ./...

t: test

build:
	@go build -o ./mnist/mnist ./cmd/mnist/

run:
	@go run ./cmd/mnist