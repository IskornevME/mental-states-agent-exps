# Mental States Agent Experiments

Minimal framework for reproducible experiments with LLM agents in interactive text environments.

The current MVP contains only the common environment/task layer for:

- ALFWorld
- ScienceWorld


## Common interaction API

Every environment implements:

```python
observation, state = env.reset()
messages = env.build_agent_messages()
observation, state = env.step(llm_output)
```

The actor is expected to answer in the shared ReAct format:

```text
<think>
reasoning
</think>
<action>
environment action
</action>
```

Admissible actions are included in actor prompts by default.

## Repository layout

```text
agent_eval/
  state.py
  paths.py
  tasks/
  envs/
  compat/
configs/
  envs/
data/
  alfworld/
  sciworld/
envs/
  scienceworld/
```

## Benchmark assets

Expected ALFWorld files:

```text
data/alfworld/base_config.yaml
data/alfworld/logic/alfred.pddl
data/alfworld/logic/alfred.twl2
data/alfworld/json_2.1.1/...
```

Expected ScienceWorld files:

```text
data/sciworld/train_indices.json
data/sciworld/dev_indices.json
data/sciworld/test_indices.json
data/sciworld/taskname2id.json
data/sciworld/max_steps.json
envs/scienceworld/scienceworld.jar
```
