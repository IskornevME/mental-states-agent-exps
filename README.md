# Mental States Agent Experiments

Minimal framework for reproducible experiments with LLM agents in interactive text environments.

The current MVP contains only the common environment/task layer for:

- ALFWorld
- ScienceWorld
- Webshop


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

### Запуск эксперимента с QNet critic

Для ALFWorld также поддерживается режим с обученным QNet critic. На каждом шаге actor генерирует несколько независимых ReAct-кандидатов из одного и того же состояния, после чего critic оценивает каждый вариант по представлению `state + action`. В среде выполняется кандидат с максимальным Q-value.

Actor и critic используют отдельные GPU: actor запускается через SGLang на `SERVER_GPU`, а QNet critic загружается локально на `CRITIC_GPU`.

Для запуска используется отдельный launcher:

```bash
scripts/run_qwen3_experiment_w_critic.sh
```

Пример:

```bash
MODEL_PATH=/home/m.iskornev/qlass/models/Qwen3-4B-Instruct-2507 \
CRITIC_MODEL_PATH=/home/m.iskornev/qlass/models/qlass-Qwen3-4B-Instruct-2507-alfworld-Q-recovered \
SERVER_GPU=5 \
CRITIC_GPU=6 \
N_CANDIDATES=2 \
N_TRAJS=3 \
RUN_ID=0 \
bash scripts/run_qwen3_experiment_w_critic.sh
```

Для короткой проверки setup можно дополнительно задать:

```bash
MAX_TASKS=2
```

Основные параметры:

* `MODEL_PATH` - checkpoint или Hugging Face model ID actor-модели.
* `CRITIC_MODEL_PATH` - путь до обученного QNet checkpoint. Параметр обязателен.
* `CRITIC_TOKENIZER_PATH` - tokenizer для critic. По умолчанию используется `MODEL_PATH`.
* `SERVER_GPU` - GPU для SGLang actor server.
* `CRITIC_GPU` - отдельная GPU для локального QNet critic.
* `N_CANDIDATES` - количество actor-кандидатов, которые critic оценивает на каждом шаге. По умолчанию `2`.
* `N_TRAJS` - количество независимых полных траекторий для каждой задачи. По умолчанию `3`.
* `MAX_TASKS` - необязательное ограничение на количество задач.
* `RUN_ID` - идентификатор запуска.
* `OUT_DIR` - позволяет явно переопределить директорию с результатами.

Сейчас QNet critic поддерживается только для ALFWorld.

По умолчанию результаты сохраняются в:

```text
outputs/qwen3_4b_alfworld_qnet_run<RUN_ID>/
```

Формат `trajectories.jsonl` совместим с обычными actor-only экспериментами. Дополнительно для каждого шага сохраняются все рассмотренные кандидаты, их Q-values, источник score и идентификатор выбранного critic'ом кандидата. Поэтому стандартный ALFWorld calculator можно использовать без изменений.



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


### Ручное прохождение задач

Для анализа среды и отдельных задач benchmark можно запустить интерактивный режим, в котором вместо LLM действия выбирает человек.

Человеку показывается тот же prompt, который в обычном эксперименте получает LLM: описание задачи, история предыдущих действий, текущий observation, inventory (для Sciworld), admissible actions и остальные инструкции среды.

В терминале необходимо вводить только действие без тегов `<think>` и `<action>`. Runner автоматически преобразует введённое действие к стандартному ReAct-формату.

По умолчанию выбираются 3 случайные задачи. Набор задач воспроизводим через `TASK_SEED`.

ALFWorld:
```bash
BENCHMARK=alfworld \
MAX_TASKS=3 \
TASK_SEED=42 \
RUN_ID=0 \
bash scripts/run_human_experiment.sh
```

### Возможные проблемы

Если launcher сообщает, что на `SGLANG_PORT` уже отвечает сервер, нужно остановить предыдущий SGLang process или выбрать другой порт:

```bash
SGLANG_PORT=21005 ...
```

Для ScienceWorld наличие `terminate_reason="negative_score"` является нормальным способом завершения episode и не означает падение программы.

## Запуск на Webshop

Есть также возможность запуска эксперимента на бенчмарке Webshop, но для этого требуется создать отдельное окружение со своими зависимостями. Для начала надо создать новое окружение и установить в него все либы из `/eval/webshop/requirements.txt`:
```bash
conda create -n qwen_webshop python=3.10
conda activate qwen_webshop
pip install -r requirements.txt
```
Затем потребуется скачать с hugging-face необходимые данные. Для этого достаточно выполнить:
```bash
python - <<'PY'
from huggingface_hub import hf_hub_download

repo_id = "YWZBrandon/webshop-data"

files = [
    "items_shuffle.json",
    "items_ins_v2.json",
    "items_human_ins.json",
]

for filename in files:
    path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
        local_dir="data",
    )
    print(f"{filename}: {path}")
PY
```
Чтобы код отработал возможно потребуется установить `huggingface_hub`.
Затем потребуется перейти в директорию `/eval/webshop/search_engine` и выполнить там команды:
```bash
python -m spacy download en_core_web_lg
mkdir -p resources resources_100 resources_1k resources_100k
conda install -c conda-forge openjdk=11
python convert_product_file_format.py
mkdir -p indexes
bash run_indexing.sh
```
После этого окружение готово к запуску эксперимента.

## Добавление нового агента
Нужно будет создать файл, в котором описать логику обращения к этому агенту:
```bash
agent_eval/agents/
    base.py
    sglang.py
    new_api.py       # новый
```
Если данный агент возвращает thinking отдельно (то есть у него есть такой режим), то в функции `act()` нового агента можно сделать что-то вроде:
```bash
def act(self, messages: List[dict]) -> str:
    response = self._call_api(messages)

    thinking = response.reasoning.strip()
    action_text = response.content.strip()

    # Если final response API уже содержит <action>, можно оставить его.
    action = parse_react_action(action_text)

    return (
        f"<think>{thinking}</think>\n"
        f"<action>{action}</action>"
    )
```
То есть лучше самим сериализовать thinking в используемый внутренний формат. Если модель имеет встроенный thinking, но он скрыт (то есть его нельзя получить), тогда возвращаем только действие `return f"<action>{action}</action>"`.

Опять же, если агент имеет встроенный thinking, то нужно будет скорретировать промпт так, чтобы в нем не просить модель еще раз размышлять. То есть надо сделать что-то вроде (для Alfworld; промты лежат в `/agent_eval/envs/alfworld.py`):
```bash
"Now it's your turn to take an action.\n"
"Choose one admissible action for the current step and present it "
"within <action> </action> tags."
```
То есть без ращзмышления. В идеале лучше добавить флаг `require_think_tags`, которые бы за это отвечал, чтоб можно было легко переключаться между агентами.

Далее в `/agent_eval/agents/__init__.py` нужно зарегестрировать нового агента:
```bash
AGENT_REGISTRY = {
    "sglang_chat": SGLangChatAgent,
    "human": HumanAgent,
    "new_api": NewAPIAgent,
}
```
И добавить новые конфиги в `/configs`:
Сначала конфиг агента в `/agents`:
```
type: new_api

model_name: some-reasoning-model

api_key_env: NEW_API_KEY
base_url: ...

max_tokens: 4096
temperature: 0.7
request_timeout: 300
```
Затем конфиг эксперимента
```
name: new_model_alfworld

agent_config: configs/agents/new_api.yaml
env_config: configs/envs/alfworld.yaml

output_dir: outputs/new_model_alfworld

max_tasks: null
num_trajectories: 3
```

Финально важно не забыть подставить нужные параметры в .sh скрипт.

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
