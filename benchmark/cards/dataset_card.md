# Dataset Card: RoCoBench-Pack-Skills-v1

## Source

Expert data for this repository is produced by RRT-backed `PUT_OBJECT_IN_CONTAINER` executions and exported through the Phase 2 LeRobot-compatible dataset path.

## Schema

Episodes follow the Phase 2 schema where `observation[t]` precedes `action[t]`. Split membership is by episode/variation, never by frame.

## Existing Debug Dataset

The repository includes a small debug dataset under `artifacts/datasets/pack_put_object_debug` and a LeRobot export under `artifacts/datasets/pack_put_object_debug_lerobot`.

## Failure Filtering

Only validated expert episodes should be used for training. Failed attempts are not silently mixed into expert data.

## Biases And Coverage

The fixed benchmark suite is intentionally small. It is suitable for release and integration checks, not broad claims about general grocery packing performance.

## License

Dataset artifacts follow the repository license unless a downstream release states otherwise.

