# Redis persistence

Redis offers two persistence options. RDB snapshots write the dataset to disk at intervals,
while the append-only file (AOF) logs every write operation so the dataset can be rebuilt on restart.

## Choosing between them

Use RDB for fast restarts and compact backups. Use AOF when you cannot afford to lose recent writes.
