# lerobot-roco-env

Python 3.12+ Gymnasium client for the Phase 0 RoCo bridge.

Install in the LeRobot/Gymnasium environment:

```bash
pip install -e "integrations/lerobot_roco/client[lerobot,test]"
```

Start the RoCo server separately from the Python 3.8 RoCo environment, then:

```python
from lerobot_roco_env import RoCoGymEnv

env = RoCoGymEnv(endpoint="tcp://127.0.0.1:5557")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.hold_action())
env.close()
```
