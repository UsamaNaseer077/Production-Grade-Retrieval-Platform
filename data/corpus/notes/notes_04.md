# Databases — Study Notes

## Overview

SQL window functions compute aggregate values over a sliding partition of rows without collapsing them. PARTITION BY groups rows; ORDER BY sorts within the partition; the frame clause specifies which rows to include in each aggregate (ROWS BETWEEN, RANGE BETWEEN). Rank, dense_rank, row_number, lead, lag, and cumulative sum are common window functions.

### Details

SQL window functions compute aggregate values over a sliding partition of rows without collapsing them. PARTITION BY groups rows; ORDER BY sorts within the partition; the frame clause specifies which rows to include in each aggregate (ROWS BETWEEN, RANGE BETWEEN).

## Key Concepts

Database normalisation eliminates redundancy by decomposing tables into relations with minimal functional dependencies. First normal form requires atomic column values. Second normal form removes partial dependencies on composite primary keys. Third normal form removes transitive dependencies. BCNF (Boyce-Codd) is a stricter variant. Over-normalisation can hurt query performance; denormalisation trades space for read speed.

### Details

Database normalisation eliminates redundancy by decomposing tables into relations with minimal functional dependencies. First normal form requires atomic column values.

## Implementation Details

LSM trees (Log-Structured Merge-Trees) write all mutations to an in-memory memtable and an append-only WAL, then periodically flush sorted runs to disk (SSTables). Background compaction merges overlapping runs to bound read amplification. LSM trees excel at write-heavy workloads; RocksDB, LevelDB, and Apache Cassandra use LSM.

### Details

LSM trees (Log-Structured Merge-Trees) write all mutations to an in-memory memtable and an append-only WAL, then periodically flush sorted runs to disk (SSTables). Background compaction merges overlapping runs to bound read amplification.

## Trade-offs and Limitations

ACID transactions guarantee Atomicity (all-or-nothing), Consistency (integrity constraints preserved), Isolation (concurrent transactions behave as if serial), and Durability (committed data survives failure). PostgreSQL implements MVCC for snapshot isolation, reducing lock contention. Serialisable isolation detects write skew anomalies via predicate locking or serialisability conflict detection graphs.

### Details

ACID transactions guarantee Atomicity (all-or-nothing), Consistency (integrity constraints preserved), Isolation (concurrent transactions behave as if serial), and Durability (committed data survives failure). PostgreSQL implements MVCC for snapshot isolation, reducing lock contention.

