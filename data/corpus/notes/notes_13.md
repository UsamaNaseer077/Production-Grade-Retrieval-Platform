# Distributed Systems — Study Notes

## Overview

Service mesh architectures (Istio, Linkerd) offload cross-cutting concerns — mutual TLS, observability, traffic management — from application code into a sidecar proxy. The control plane configures proxies centrally; the data plane intercepts all traffic. This enables canary deployments, circuit breakers, and rate limiting without code changes.

### Details

Service mesh architectures (Istio, Linkerd) offload cross-cutting concerns — mutual TLS, observability, traffic management — from application code into a sidecar proxy. The control plane configures proxies centrally; the data plane intercepts all traffic.

## Key Concepts

Raft consensus achieves fault-tolerant distributed agreement through leader election, log replication, and safety properties. A leader is elected by majority vote; entries are committed only after replication to a majority of nodes. Followers that fall behind re-sync by requesting missing log entries. etcd and TiKV use Raft for distributed coordination.

### Details

Raft consensus achieves fault-tolerant distributed agreement through leader election, log replication, and safety properties. A leader is elected by majority vote; entries are committed only after replication to a majority of nodes.

## Implementation Details

Distributed tracing instruments microservice calls with span context propagated via HTTP headers (W3C TraceContext standard). Each service creates child spans linked to the parent. Aggregated traces reveal end-to-end latency breakdown, identify bottlenecks, and surface cascading failures invisible to per-service metrics. Jaeger and Zipkin are popular backends.

### Details

Distributed tracing instruments microservice calls with span context propagated via HTTP headers (W3C TraceContext standard). Each service creates child spans linked to the parent.

## Trade-offs and Limitations

The CAP theorem states that a distributed system can provide at most two of: Consistency (every read receives the most recent write), Availability (every request receives a response), and Partition tolerance (the system continues operating despite network partitions). In practice, partitions are unavoidable in WANs so systems trade off between CP (HBase, Zookeeper) and AP (Cassandra, CouchDB).

