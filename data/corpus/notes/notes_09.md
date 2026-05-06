# Software Engineering — Study Notes

## Overview

Observability in production requires three pillars: metrics (time-series counters/gauges via Prometheus), logs (structured JSON via ELK or Loki), and traces (distributed spans via Jaeger). The USE method (Utilisation, Saturation, Errors) guides resource analysis; the RED method (Rate, Errors, Duration) guides service health. SLOs bound acceptable latency percentiles and error rates.

### Details

Observability in production requires three pillars: metrics (time-series counters/gauges via Prometheus), logs (structured JSON via ELK or Loki), and traces (distributed spans via Jaeger). The USE method (Utilisation, Saturation, Errors) guides resource analysis; the RED method (Rate, Errors, Duration) guides service health.

## Key Concepts

Continuous integration runs automated builds and tests on every commit to catch integration failures early. GitHub Actions, GitLab CI, and Jenkins define workflows as YAML pipelines. Trunk-based development merges to main frequently in small increments; feature flags decouple deployment from release. Branch protection rules enforce CI passes before merging.

### Details

Continuous integration runs automated builds and tests on every commit to catch integration failures early. GitHub Actions, GitLab CI, and Jenkins define workflows as YAML pipelines.

## Implementation Details

Test-driven development (TDD) writes tests before implementation: red (failing test), green (minimal implementation), refactor (clean code). Tests serve as living documentation and prevent regression. Property-based testing (Hypothesis, QuickCheck) generates random inputs to find edge cases that hand-written tests miss. Mutation testing checks test suite quality by introducing deliberate bugs.

### Details

Test-driven development (TDD) writes tests before implementation: red (failing test), green (minimal implementation), refactor (clean code). Tests serve as living documentation and prevent regression.

## Trade-offs and Limitations

Clean Architecture organises code in concentric rings: entities (business rules), use cases (application logic), interface adapters (controllers/presenters), and infrastructure (databases, frameworks). Dependencies point inward; outer layers depend on inner abstractions not implementations. This decouples business logic from framework details, enabling independent testing of each layer.

### Details

Clean Architecture organises code in concentric rings: entities (business rules), use cases (application logic), interface adapters (controllers/presenters), and infrastructure (databases, frameworks). Dependencies point inward; outer layers depend on inner abstractions not implementations.

