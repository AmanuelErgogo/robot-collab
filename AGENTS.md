# AGENTS.md

## Repository purpose

This repository implements RoCo and RoCoBench, a Python 3.8 MuJoCo/dm_control
multi-robot collaboration system.

## Phase 0 objective

Implement a compatibility bridge allowing a Python 3.12+ Gymnasium/LeRobot
client to control a Python 3.8 RoCoBench simulator process.

## Runtime separation

- RoCo runtime remains Python 3.8.
- Current LeRobot client runs in Python 3.12+.
- Do not install LeRobot into the RoCo environment.
- Do not import `rocobench` from the client package.
- Do not import Gymnasium or LeRobot from normal RoCo modules.
- Communicate through the versioned local protocol only.

## Architecture rules

- Simulator state is owned by the server.
- Gym episode state is mirrored and validated by the client.
- Derive action/state dimensions from runtime metadata.
- Use `SimAction` and existing `env.step()` semantics.
- Keep one active agent and hold passive agents in Phase 0.
- Use LeRobot-standard raw keys `pixels` and `agent_pos`.
- Always return `info["is_success"]`.
- Keep protocol, serialization, simulator adapters, and Gym wrapper separate.

## Security

- No pickle over RPC.
- No `eval` or `exec`.
- Bind to localhost by default.
- Validate protocol version, payload size, ndarray dtype/shape, action bounds,
  episode ID, and step index.
- Disable debug commands and remote shutdown by default.
- Do not log binary arrays, tokens, or secrets.

## Compatibility

- Preserve Python 3.8 syntax in shared/server modules.
- Client package may use Python 3.12 syntax only inside the isolated client
  package.
- Do not modify legacy LLM behavior.
- Avoid unrelated refactors.

## Verification

Run server/common tests in the RoCo environment and client/LeRobot tests in the
Python 3.12 environment. Do not claim MuJoCo, rendering, Gym checker, or LeRobot
integration passed unless each command actually ran.
