# Сбор данных для обучения QNet critic

В этой директории находится минимальный pipeline для получения обучающего датасета QNet critic.

Pipeline основан на алгоритме сбора данных из QLASS, но использует текущие `Task`, `Env` и `Agent` интерфейсы репозитория и общий single-message ReAct-формат.

Поддерживаются:

* ALFWorld;
* ScienceWorld;
* WebShop.

## Общая схема

Процесс состоит из двух независимых этапов:

```text
expert trajectories
        +
выбранный actor
        +
benchmark
        ↓
QLASS-style tree exploration
        ↓
worker_*_trees.jsonl
        ↓
backward Q propagation
        ↓
q_dataset.json
        ↓
QNet critic training
```

Код обучения QNet critic находится вне этой директории.

## Важное замечание

Данные для critic являются **benchmark-specific** и **actor-specific**.

Например, если предполагается использовать critic совместно с определённой версией Qwen actor на ScienceWorld, желательно собирать данные именно:

* на ScienceWorld;
* этим actor;
* с тем же ReAct-format;
* с теми же основными настройками prompt/environment.

Critic, обученный на trajectories одного actor, не гарантированно будет полезен другому actor, поскольку распределение посещаемых состояний и выбираемых действий может существенно отличаться.

## Expert trajectories

В репозитории используются:

```text
data/expert/alfworld_sft.json
data/expert/sciworld_sft.json
data/expert/webshop_sft.json
```

Эти trajectories находятся в legacy QLASS-format (`Thought: ... / Action: ...`).

Во время collection они автоматически преобразуются в текущий формат:

```text
<think>...</think>
<action>...</action>
```

Старые state prompts из expert trajectories при этом **не используются**.


## Алгоритм exploration

Для каждой задачи строится дерево.

### 1. Actor exploration

Начиная с root state, actor генерирует несколько альтернативных действий:

```text
SAMPLES_PER_DEPTH
```

Для получения разных вариантов при генерации второго и последующих кандидатов actor временно получает дополнительную инструкцию с просьбой предложить другой ответ.

Эта дополнительная инструкция используется **только для sampling** и не сохраняется в `critic_state`. После каждого нового действия policy выполняет rollout до terminal state.

### 2. Expansion rollout branches

Если terminal reward rollout выше:

```text
POSITIVE_REWARD_THRESHOLD
```

узлы этой rollout-ветки после `MIN_PRUNE_DEPTH` могут быть дополнительно поставлены в очередь для дальнейшего expansion.

`MAX_DEPTH` ограничивает глубину повторного expansion. При этом rollout сам по себе не обрезается на `MAX_DEPTH`: он продолжается до нормального terminal state или benchmark `MAX_STEPS`.

### 3. Expert path

После actor exploration в то же дерево добавляется expert trajectory.

Кроме того, начиная со второго expert state и до `MAX_DEPTH`, actor генерирует одну альтернативную ветку (`brother branch`), которая также rollout'ится до terminal state.

В результате дерево содержит одновременно:

```text
exploration branches
+
policy rollouts
+
expert path
+
brother rollouts around expert path
```

## Параметры по умолчанию

Reference-настройки:

```text
MAX_DEPTH=5
MIN_PRUNE_DEPTH=3
SAMPLES_PER_DEPTH=2
POSITIVE_REWARD_THRESHOLD=0.01
HISTORY_LENGTH=50
NUM_WORKERS=8
```

Для ALFWorld reference collection используется:

```text
MAX_STEPS=30
```

Для остальных benchmark launcher по умолчанию использует:

```text
ScienceWorld: MAX_STEPS=40
WebShop:      MAX_STEPS=5
```

Все значения можно переопределить через environment variables.

## Запуск smoke test

Перед полным collection рекомендуется проверить pipeline на двух задачах:

```bash
BENCHMARK=alfworld \
MODEL_PATH=/path/to/Qwen3-4B-Instruct-2507 \
SERVER_GPU=5 \
SMOKE_TEST=1 \
RUN_ID=smoke \
bash critic_data/run_collect_sglang.sh
```

В smoke mode автоматически используются:

```text
NUM_WORKERS=1
MAX_TASKS_PER_WORKER=2
```

После успешного запуска появится:

```text
outputs/critic_data/alfworld/runsmoke/
├── worker_0_trees.jsonl
└── logs/
    ├── sglang_server.log
    └── worker_0.log
```

## Полный collection

Пример для ALFWorld:

```bash
BENCHMARK=alfworld \
MODEL_PATH=/path/to/Qwen3-4B-Instruct-2507 \
SERVER_GPU=5 \
NUM_WORKERS=8 \
RUN_ID=0 \
bash critic_data/run_collect_sglang.sh
```

ScienceWorld:

```bash
BENCHMARK=sciworld \
MODEL_PATH=/path/to/Qwen3-4B-Instruct-2507 \
SERVER_GPU=5 \
NUM_WORKERS=8 \
RUN_ID=0 \
bash critic_data/run_collect_sglang.sh
```

WebShop:

```bash
BENCHMARK=webshop \
MODEL_PATH=/path/to/Qwen3-4B-Instruct-2507 \
SERVER_GPU=5 \
NUM_WORKERS=8 \
RUN_ID=0 \
bash critic_data/run_collect_sglang.sh
```

## Использование другого actor

Actor задаётся через:

```text
AGENT_CONFIG
MODEL_PATH
```

Например:

```bash
BENCHMARK=alfworld \
AGENT_CONFIG=configs/agents/my_agent.yaml \
MODEL_PATH=/path/to/my/model \
SERVER_GPU=5 \
RUN_ID=my_agent \
bash critic_data/run_collect_sglang.sh
```

Collector использует `AGENT_REGISTRY` из основного репозитория.

Поэтому новый actor, добавленный в стандартную инфраструктуру репозитория, можно использовать для collection без изменения tree-search algorithm.

## Основные параметры collection

### `BENCHMARK`

Допустимые значения:

```text
alfworld
sciworld
webshop
```

### `NUM_WORKERS`

Количество параллельных collector processes.

По умолчанию:

```text
8
```

Все workers обращаются к одному SGLang server.

### `MAX_DEPTH`

Максимальная глубина повторного expansion дерева.

По умолчанию:

```text
5
```

Этот параметр не ограничивает длину rollout до terminal state.

### `MIN_PRUNE_DEPTH`

Минимальная глубина, после которой узлы положительной rollout-ветки могут быть поставлены обратно в очередь для дальнейшего expansion.

По умолчанию:

```text
3
```

### `SAMPLES_PER_DEPTH`

Желаемое количество альтернативных child actions на расширяемом state.

По умолчанию:

```text
2
```

### `POSITIVE_REWARD_THRESHOLD`

Rollout считается достаточно положительным для дополнительного expansion, если:

```text
terminal_reward > POSITIVE_REWARD_THRESHOLD
```

По умолчанию:

```text
0.01
```

Это значение сохранено из QLASS.

Для benchmark с другим reward scale его можно переопределить.

### `HISTORY_LENGTH`

Количество предыдущих ReAct interaction steps, которое environment включает в actor state.

По умолчанию:

```text
50
```

### `MAX_STEPS`

Максимальная длина episode.

Можно явно переопределить, например:

```bash
MAX_STEPS=30
```

### `MAX_TASKS_PER_WORKER`

Необязательное ограничение числа задач для каждого worker.

Например:

```bash
MAX_TASKS_PER_WORKER=10
```

### `OVERWRITE`

По умолчанию collector отказывается перезаписывать существующий worker tree.

Чтобы явно разрешить перезапись:

```bash
OVERWRITE=1
```

## Формат дерева

Каждый worker сохраняет отдельный JSONL:

```text
worker_0_trees.jsonl
worker_1_trees.jsonl
...
```

Одна строка соответствует одной benchmark task.

Пример верхнего уровня:

```json
{
  "benchmark": "alfworld",
  "task_id": "...",
  "actor": {
    "type": "sglang_chat",
    "model_name": "...",
    "agent_config": "configs/agents/qwen3_4b.yaml"
  },
  "search": {
    "max_depth": 5,
    "min_prune_depth": 3,
    "samples_per_depth": 2
  },
  "expert": {
    "num_steps": 8,
    "reward": 1.0,
    "success": true
  },
  "root": {}
}
```

Каждый не-root tree node содержит:

```json
{
  "critic_state": "...",
  "critic_action": "...",
  "reward": 0.0,
  "finished": false,
  "success": false,
  "source": "exploration",
  "children": []
}
```

`source` может быть:

```text
exploration
rollout
expert
expert_brother
```

## Построение Q dataset

После завершения collection необходимо преобразовать деревья в QNet training samples.

Например:

```bash
DATA_DIR=outputs/critic_data/alfworld/run0 \
bash critic_data/run_build_dataset.sh
```

По умолчанию используются:

```text
GAMMA=0.9
UPPER_NUM=300
SEED=42
```

## Расчет Q-value

Используется `vanilla` target из QLASS.

Перед расчётом Q:

* reward всех internal tree nodes устанавливается в `0`;
* terminal leaves сохраняют environment reward.

После этого:

```text
leaf:
Q = reward
```

и:

```text
internal node:
Q = reward + gamma * max(Q(child))
```

Поскольку internal reward предварительно обнуляется:

```text
Q = gamma * max(Q(child))
```

Для каждой benchmark task сохраняется не более:

```text
UPPER_NUM
```

случайно выбранных examples.

После sampling Q-values min-max нормализуются **отдельно внутри каждой task**, как в reference QLASS pipeline.

## Формат итогового dataset

По умолчанию создаётся:

```text
<DATA_DIR>/q_dataset.json
```

Каждый example имеет вид:

```json
{
  "id": "task_id",
  "conversations": [
    {
      "from": "human",
      "value": "critic_state"
    },
    {
      "from": "gpt",
      "value": "critic_action"
    }
  ],
  "label": 0.73
}
```

Именно этот формат будет использоваться на следующем этапе для обучения Qwen3-4B QNet critic.
