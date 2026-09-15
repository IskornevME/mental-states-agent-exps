# Обучение QNet critic

В этой директории находится минимальный pipeline для обучения QNet critic на данных, полученных через `critic_data/`.

Training-код является benchmark-agnostic и не содержит логики ALFWorld, ScienceWorld или WebShop. Benchmark и actor уже отражены в собранном `q_dataset.json`.

## Общая схема

Полный pipeline выглядит следующим образом:

```text
actor + benchmark
        ↓
critic_data/collect.py
        ↓
search trees
        ↓
critic_data/build_dataset.py
        ↓
q_dataset.json
        ↓
critic_training/train.py
        ↓
trained QNet checkpoint
        ↓
agent_eval/critics/qnet.py
```

## Архитектура QNet

QNet состоит из:

```text
pretrained causal-LM backbone
        ↓
hidden state последнего непаддингового токена
        ↓
Linear(hidden_size, 1024)
ReLU
Linear(1024, 1024)
ReLU
Linear(1024, 1)
        ↓
scalar Q(s, a)
```

На вход critic получает:

```text
user      → critic_state
assistant → critic_action
```

То есть один training example соответствует конкретной паре `(state, action)`.

Loss:

```text
MSE(predicted_Q, target_Q)
```

## Backbone

Training-код не привязан к Qwen3-4B.

В качестве backbone может использоваться стандартная HuggingFace decoder-only causal language model, совместимая с:

```python
AutoModelForCausalLM
```

и имеющая tokenizer с `chat_template`.

Например, это может быть Qwen, Llama или другая совместимая instruct/chat model.

Важно отличать две вещи:

* **actor**, которым собирались trajectories;
* **backbone critic**, который обучается оценивать эти trajectories.

Они не обязаны быть одной и той же моделью.

Например:

```text
actor:          Qwen3-4B
critic backbone: Llama-based model
```

технически допустим.

При этом critic-data желательно собирать именно actor'ом, которым впоследствии будут управлять: распределение состояний и действий зависит от policy actor'а.

## Training data

Training dataset должен быть предварительно построен:

```bash
DATA_DIR=outputs/critic_data/alfworld/run0 \
bash critic_data/run_build_dataset.sh
```

Результат:

```text
outputs/critic_data/alfworld/run0/q_dataset.json
```

Каждый example имеет формат:

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

`label` - нормализованный Q-value, рассчитанный в `critic_data/build_dataset.py`.

## Установка зависимостей


```bash
pip install -r critic_training/requirements_alfworld_qwen.txt
```

## Базовый запуск

Необходимо задать:

```text
BACKBONE_PATH
Q_DATA_PATH
```

Например:

```bash
BACKBONE_PATH=/path/to/Qwen3-4B-Instruct-2507 \
Q_DATA_PATH=outputs/critic_data/alfworld/run0/q_dataset.json \
GPU_LIST=0,1 \
RUN_NAME=qwen3_4b_alfworld_qnet \
bash critic_training/run_train.sh
```

По умолчанию checkpoint будет сохранён в:

```text
outputs/critic_training/qwen3_4b_alfworld_qnet/
```

## Запуск с другим backbone

Например:

```bash
BACKBONE_PATH=/path/to/another-chat-model \
Q_DATA_PATH=outputs/critic_data/sciworld/run0/q_dataset.json \
GPU_LIST=0,1 \
RUN_NAME=my_sciworld_qnet \
bash critic_training/run_train.sh
```

Никаких изменений в `train.py` для другого backbone делать не требуется, если модель совместима с `AutoModelForCausalLM` и её tokenizer имеет `chat_template`.

## Основные параметры

Reference defaults сохранены из QLASS:

```text
GLOBAL_BATCH_SIZE=64
MICRO_BATCH_SIZE=1
MODEL_MAX_LENGTH=8192
NUM_EPOCHS=2
LEARNING_RATE=1e-5
WARMUP_RATIO=0.03
WEIGHT_DECAY=0.0
SEED=42
```

Их можно переопределять через environment variables.

Например:

```bash
GLOBAL_BATCH_SIZE=32 \
MICRO_BATCH_SIZE=1 \
NUM_EPOCHS=3 \
LEARNING_RATE=5e-6 \
...
```

## Sequence length

Training использует:

```text
padding_side = right
truncation_side = left
```

Это важно. QNet оценивает hidden state последнего непаддингового токена, поэтому right padding должен совпадать с inference.

Left truncation позволяет при слишком длинном prompt сохранить наиболее свежую часть trajectory и candidate action в конце последовательности.

`MODEL_MAX_LENGTH` должен совпадать с `max_prompt_tokens`, который затем используется в critic inference config.

По умолчанию:

```text
8192
```

## Chat template

Перед tokenization:

```python
tokenizer.apply_chat_template(
    [
        {"role": "user", "content": critic_state},
        {"role": "assistant", "content": critic_action},
    ],
    tokenize=False,
    add_generation_prompt=False,
)
```

После этого tokenizer вызывается с:

```text
add_special_tokens=False
```

Это должно совпадать с `agent_eval/critics/qnet.py`.

## Multi-GPU training

При использовании более одной GPU launcher автоматически включает FSDP:

```text
full_shard + auto_wrap
```

Например:

```bash
GPU_LIST=0,1 \
...
bash critic_training/run_train.sh
```

`train.py` автоматически читает `_no_split_modules` у выбранного HuggingFace backbone и передает соответствующие layer classes в FSDP.

Поэтому один launcher можно использовать для разных архитектур.

## Single-GPU training

При одной GPU FSDP не используется:

```bash
GPU_LIST=0 \
...
bash critic_training/run_train.sh
```

Reference QLASS training обучает весь backbone, а не только MLP head; новое обучение сохраняет это поведение.

## Smoke test

Для проверки pipeline без полного обучения можно ограничить обучение двумя optimizer steps:

```bash
BACKBONE_PATH=/path/to/model \
Q_DATA_PATH=/path/to/q_dataset.json \
GPU_LIST=0,1 \
TRAIN_MAX_STEPS=2 \
RUN_NAME=smoke_qnet \
bash critic_training/run_train.sh
```

После этого необходимо проверить наличие:

```text
outputs/critic_training/smoke_qnet/
├── config.json
├── pytorch_model.bin
├── qnet_training_config.json
├── tokenizer_config.json
├── trainer_state.json
└── ...
```

## Формат checkpoint

Главные файлы:

```text
pytorch_model.bin
config.json
tokenizer files
qnet_training_config.json
```

`pytorch_model.bin` содержит одновременно:

```text
llama.*  → trained causal-LM backbone
mlp.*    → QNet regression head
```

Название `llama` сохранено ради совместимости со старыми QLASS checkpoint'ами; оно не означает, что backbone обязан быть Llama.

## Resume training

Если `OUTPUT_DIR` содержит HuggingFace checkpoint вида:

```text
checkpoint-1000/
```

training автоматически продолжится с последнего checkpoint.

Если директория непустая, но resumable checkpoint отсутствует, training остановится, чтобы случайно не перезаписать существующую модель.

Для намеренного нового запуска следует выбрать другой `RUN_NAME` / `OUTPUT_DIR`.

## Sigmoid

Reference setup использует:

```text
APPLY_SIGMOID=False
```

Если sigmoid включается при training:

```bash
APPLY_SIGMOID=True
```

то при inference в critic config также обязательно должно быть:

```yaml
apply_sigmoid: true
```

Training и inference должны использовать одинаковую настройку.

## Использование обученного critic

После обучения checkpoint можно использовать напрямую:

```bash
MODEL_PATH=/path/to/actor \
CRITIC_MODEL_PATH=outputs/critic_training/my_qnet \
BENCHMARK=alfworld \
SERVER_GPU=5 \
CRITIC_GPU=6 \
bash scripts/run_qwen3_experiment_w_critic.sh
```

По умолчанию tokenizer critic теперь также берётся из:

```text
CRITIC_MODEL_PATH
```

При необходимости его можно переопределить:

```bash
CRITIC_TOKENIZER_PATH=/path/to/tokenizer
```

## Важные условия совместимости

Для корректного critic inference должны совпадать:

```text
training MODEL_MAX_LENGTH
    |
critic max_prompt_tokens
```

и:

```text
training APPLY_SIGMOID
    |
critic apply_sigmoid
```

Также critic должен обучаться на `q_dataset.json`, собранном для нужного benchmark и желательно нужного actor.
