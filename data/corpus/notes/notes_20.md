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

Docker packages applications with their dependencies into portable images layered on a union filesystem. Dockerfile instructions create image layers; each RUN command adds a layer. Multi-stage builds compile in a build container and copy only runtime artifacts to the final image, minimising image size. Docker Compose orchestrates multi-container apps locally.

### Details

Docker packages applications with their dependencies into portable images layered on a union filesystem. Dockerfile instructions create image layers; each RUN command adds a layer.

## Trade-offs and Limitations

Clean Architecture organises code in concentric rings: entities (business rules), use cases (application logic), interface adapters (controllers/presenters), and infrastructure (databases, frameworks). Dependencies point inward; outer layers depend on inner abstractions not implementations. This decouples business logic from framework details, enabling independent testing of each layer.

### Details

Clean Architecture organises code in concentric rings: entities (business rules), use cases (application logic), interface adapters (controllers/presenters), and infrastructure (databases, frameworks). Dependencies point inward; outer layers depend on inner abstractions not implementations.

