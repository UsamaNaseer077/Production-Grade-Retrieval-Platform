# Distributed Systems — Study Notes

## Overview

Service mesh architectures (Istio, Linkerd) offload cross-cutting concerns — mutual TLS, observability, traffic management — from application code into a sidecar proxy. The control plane configures proxies centrally; the data plane intercepts all traffic. This enables canary deployments, circuit breakers, and rate limiting without code changes.

### Details

Service mesh architectures (Istio, Linkerd) offload cross-cutting concerns — mutual TLS, observability, traffic management — from application code into a sidecar proxy. The control plane configures proxies centrally; the data plane intercepts all traffic.

## Key Concepts

Consistent hashing distributes keys across nodes in a ring topology so that only K/N keys move when a node joins or leaves (K=keys, N=nodes). Virtual nodes (vnodes) improve load balance by placing each physical node at multiple positions in the ring. Cassandra, Riak, and Amazon DynamoDB use consistent hashing for horizontal scalability without a centralised coordinator.

### Details

Consistent hashing distributes keys across nodes in a ring topology so that only K/N keys move when a node joins or leaves (K=keys, N=nodes). Virtual nodes (vnodes) improve load balance by placing each physical node at multiple positions in the ring.

## Implementation Details

Distributed tracing instruments microservice calls with span context propagated via HTTP headers (W3C TraceContext standard). Each service creates child spans linked to the parent. Aggregated traces reveal end-to-end latency breakdown, identify bottlenecks, and surface cascading failures invisible to per-service metrics. Jaeger and Zipkin are popular backends.

### Details

Distributed tracing instruments microservice calls with span context propagated via HTTP headers (W3C TraceContext standard). Each service creates child spans linked to the parent.

## Trade-offs and Limitations

Event sourcing stores state as an append-only log of events rather than mutable records. The current state is derived by replaying all events from the beginning or from a snapshot. CQRS (Command Query Responsibility Segregation) separates the write model (commands, events) from the read model (projections, queries). Kafka is commonly used as the durable event log.

### Details

Event sourcing stores state as an append-only log of events rather than mutable records. The current state is derived by replaying all events from the beginning or from a snapshot.

