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