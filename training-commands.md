```




python examples/train_simultaneous.py --size 5 --fog --self-play \
      --resume models/size5/stage4/ppo_stage3_<LATEST>_steps.zip \
      --start-opponent-phase 6 --timesteps 10M --warmup-steps 0 \
      --learning-rate 1e-4 --ent-coef 0.1 --n-steps 4096 \
      --advance-threshold 0.75 --backtrack-threshold 0.45 \
      --min-steps-per-level 50k --pool-size 20\
      --discord-webhook "$DISCORD_WEBHOOK" --discord-check-in 30


``` 

```
```
