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

Скрипты `calc_results_sciworld.py` и `calc_results_alfworld.py` нужны для подсчета итоговых метрик. Пример запуска:
```bash
python scripts/calc_results_alfworld.py --input "outputs/qwen3_4b_alfworld_run0" --output "outputs/qwen3_4b_alfworld_run0/metrics.json"
```


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
