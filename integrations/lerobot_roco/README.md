# Phase 0 RoCoBench to Gymnasium/LeRobot Bridge

This integration keeps the legacy RoCo simulator process separate from the
Gymnasium/LeRobot client process.

```text
Python 3.8 RoCo runtime
  PackGroceryTask, MuJoCo, dm_control, SimAction

localhost protocol

Python 3.12+ client runtime
  Gymnasium RoCoGymEnv, future LeRobot preprocessing/policy inference
```

Start the server in the RoCo environment:

```bash
python scripts/start_roco_bridge.py --task pack --active-agent Alice --headless
```

Run the client smoke test in the Gymnasium/LeRobot environment:

```bash
python scripts/smoke_test_roco_bridge.py --steps 5
```

The bridge protocol is versioned (`0.1`) and transfers NumPy arrays through an
explicit safe encoding. It does not use pickle over RPC.

For full setup, install, test, and demo commands, see
[`docs/phase0_setup_install_test_demos.md`](../../docs/phase0_setup_install_test_demos.md).
