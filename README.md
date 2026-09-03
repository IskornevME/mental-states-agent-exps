# Mental States Agent Experiments

Minimal framework for reproducible experiments with LLM agents in interactive text environments.

The current MVP contains only the common environment/task layer for:

- ALFWorld
- ScienceWorld


## Инструкция по запуску
1. Необходимо склонировать репозиторий и зайти в корень:
```bash
cd mental-states-agent-exps
```
2. Затем создать окружение (я делал через conda) и установить в него зависимости из файла `requirements_qwen_alfworld_sciworld.txt`:
```bash
conda create -n qwen_alfworld_sciworld python=3.10
conda activate qwen_alfworld_sciworld
conda install pip
pip install -r requirements_qwen_alfworld_sciworld.txt
```
Если вы планируете проводить эксперименты на Sciworld, то дополнительно нужно поставить java:
```bash
conda install -c conda-forge openjdk=17
```
3. Далее нужно скачать необходимые файлы для сред Alfworld/Sciworld  
Для Sciworld нужно скачать `scienceworld.jar` и положить в `envs/scienceworld/scienceworld.jar`. Скачать можно отсюда: https://drive.google.com/file/d/1dnD6qJzsJcJ2npmQowIOUiUE-tGuB-GX/view?usp=sharing  
Для Alfowrld нужно скачать архив `json_2.1.1.tar.gz`, положить в `data/alfworld/` и распаковать. Скачать отсюда: https://drive.google.com/file/d/1qfJgpMKrpZQMJG354JD8HLmzH5quzDP6/view?usp=sharing
4. Я проводил эксперименты в основном на `Qwen3-4B-Instruct-2507` (https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), поэтому все гиперпараметры генерации сейчас подобраны именно для это модели. При запуске скрипта важно передать путь до модели через переменную `MODEL_PATH`
5. Для запуска эксперимента нужно перейти в корень (`mental-states-agent-exps`) и запустить специальный bash скрипт `scripts/run_qwen3_experiment.sh`. Он поднимет sglang сервер с моделью и запустит сам эксперимент (`run_experiment.py`). Пример команды для alfworld:
```bash
MODEL_PATH=/home/m.iskornev/qlass/models/Qwen3-4B-Instruct-2507 BENCHMARK=alfworld SERVER_GPU=5 RUN_ID=0 bash scripts/run_qwen3_experiment.sh
```
Для начала для отладки можно запустить скрипт на небольшом семпле задач. Для этого можно добавить параметр MAX_TASKS.


### Основные параметры запуска

Основные настройки эксперимента передаются через переменные окружения:

* `MODEL_PATH` - путь до локального checkpoint модели или Hugging Face model ID.
* `BENCHMARK` - среда для запуска: `alfworld` или `sciworld`.
* `SERVER_GPU` - GPU, на которой будет поднят SGLang server.
* `N_TRAJS` - количество независимых траекторий для каждой задачи. По умолчанию `3`.
* `MAX_TASKS` - ограничение на количество задач. Если параметр не задан, запускается весь выбранный split.
* `RUN_ID` - идентификатор запуска, используемый в имени output-директории.
* `OUT_DIR` - при необходимости позволяет явно переопределить директорию с результатами.
* `SGLANG_PORT` - порт SGLang server. По умолчанию `21003`.
* `CONTEXT_LENGTH` - размер контекста модели. По умолчанию `32768`.

Постоянные настройки агента, среды и эксперимента находятся в:

```text
configs/agents/
configs/envs/
configs/experiments/
```

Например, `configs/envs/alfworld.yaml` задает `max_steps`, длину истории и использование admissible actions.

### Быстрая проверка установки

Перед полным экспериментом рекомендуется проверить setup на двух задачах.

ALFWorld:

```bash
MODEL_PATH=/path/to/Qwen3-4B-Instruct-2507 \
BENCHMARK=alfworld \
SERVER_GPU=0 \
MAX_TASKS=2 \
RUN_ID=smoke_alfworld \
bash scripts/run_qwen3_experiment.sh
```

ScienceWorld:

```bash
MODEL_PATH=/path/to/Qwen3-4B-Instruct-2507 \
BENCHMARK=sciworld \
SERVER_GPU=0 \
MAX_TASKS=2 \
RUN_ID=smoke_sciworld \
bash scripts/run_qwen3_experiment.sh
```


### Результаты эксперимента

По умолчанию launcher сохраняет результаты в:

```text
outputs/qwen3_4b_<benchmark>_run<RUN_ID>/
```

После успешного запуска там находятся:

```text
trajectories.jsonl   # полные траектории агента
metrics.json         # итоговые benchmark-метрики
run_metadata.json    # основные параметры и статус запуска
logs/
    sglang_server.log
    experiment.log
```

`trajectories.jsonl` содержит по одной траектории на строку. Для каждого шага сохраняются observation, prompt агента, admissible actions, полный ответ модели с `<think>` и `<action>`, распарсенное действие и следующий observation.

Launcher автоматически запускает подсчёт метрик после завершения эксперимента. При необходимости их можно пересчитать отдельно:

```bash
python scripts/calc_results_alfworld.py \
    --input outputs/qwen3_4b_alfworld_run0 \
    --output outputs/qwen3_4b_alfworld_run0/metrics.json
```

или:

```bash
python scripts/calc_results_sciworld.py \
    --input outputs/qwen3_4b_sciworld_run0 \
    --output outputs/qwen3_4b_sciworld_run0/metrics.json
```

### Повторный запуск

`run_experiment.py` по умолчанию не перезаписывает существующий `trajectories.jsonl`. Поэтому для нового эксперимента лучше использовать новый `RUN_ID`:

```bash
RUN_ID=1 ...
```

Если запуск завершился уже после сохранения trajectories, но упал только подсчёт метрик, повторно запускать inference не нужно - достаточно отдельно запустить соответствующий `calc_results_*.py`.

### Возможные проблемы

Если launcher сообщает, что на `SGLANG_PORT` уже отвечает сервер, нужно остановить предыдущий SGLang process или выбрать другой порт:

```bash
SGLANG_PORT=21005 ...
```

Для ScienceWorld наличие `terminate_reason="negative_score"` является нормальным способом завершения episode и не означает падение программы.



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
