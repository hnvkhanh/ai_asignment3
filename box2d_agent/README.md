# Box2D CarRacing PPO

Train:

```powershell
python box2d_agent/train_ppo_carracing.py --gpu-id 0
```

Watch a trained checkpoint:

```powershell
python box2d_agent/train_ppo_carracing.py `
  --eval-only `
  --checkpoint runs_box2d/CarRacing-v3_YYYYMMDD_HHMMSS/ppo_carracing_final.pt `
  --render-mode human
```

CarRacing uses a continuous action space, so this agent uses PPO with a CNN
actor-critic instead of DQN.
