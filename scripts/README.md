## Заметки для оценки актора

Пусть expert trajectory такая:

```
S0 --A0*--> S1 --A1*--> S2 --A2*--> S3
```

где:

- `S0` — начальное состояние;
- `A0*` — expert action;
- `S1` — состояние, полученное после `A0*`;
- `A1*` — следующее expert action;
- и т. д.

Мы хотим проверить:

> Какое действие выбрал бы наш Qwen actor, если поставить его именно в `S0`, `S1`, `S2`, ...?
>

Шаг 1. Берем конкретную expert trajectory

С помощью ее game file идентифицируем конкретную Aflworld среду (под данную задачу).
game_file лежит в `/mental-states-agent-exps/data/alfworld/json_2.1.1`

Шаг 2. Создаем соответствующую ALFWorld environment

Шаг 3. На `S0` строим настоящий prompt Actor

`messages = env.build_agent_messages()`

Шаг 4. Спрашиваем Actor

`actor_output = actor.act(messages)`
Получаем что-то вроде:
```
<think>
I should inspect the countertop.
</think>
<action>
go to countertop 1
</action>
```

Парсим отсюда действие

Шаг 5. Сравниваем с expert action

Достаем экспертное действие - `expert_action = steps[0]["action"]`  
Если действия совпали - считаем что успех, иначе - неудача

Шаг 6. Самый важный момент: НЕ выполняем Actor action

Выполняем именно expert_action, то есть:
`env.step(steps[0]["react_output"])`

В результате придем именно в S1 из экспертной траектории

Шаг 7. Теперь оцениваем Actor уже на `S1`

Снова `messages = env.build_agent_messages()`  
И теперь сравниваем его с `steps[1]["action"]`  и тд

### В результате

Получается:

```
reset()

S0
 |
 | ask Actor
 | compare with A0*
 |
 | execute A0*
 v
S1
 |
 | ask Actor
 | compare with A1*
 |
 | execute A1*
 v
S2
 |
 | ask Actor
 | compare with A2*
 |
 | execute A2*
 v
...
```

Кажется лучше пока реализовать отдельный **скрипт для Actor evaluation** рядом с run_experiments.py. Возможно потребуется еще внести изменения в /agent_eval/envs/alfworld.py. Сам run_experiments.py можно использовать в качестве референса.

Экспертные траектории лежат в `/mental-states-agent-exps/data/expert/`  
Скрипт-пример взаимодействия с траекториями - `mental-states-agent-exps/scripts/inspect_alfworld_expert_trajectories.py`

Для того, чтобы оценить актора через **LLM-as-a-Judge** надо будет на каждом шаге отправлять действие, которое генерирует актор (`raw_output = actor.act(messages)`) вметсе с дополнительным контекстом (описанием задачи, историей, текущим стейтом, мб чем-то еще) на вход LLM и спрашивать что-то вроде "является ли действие актора правильным в данных условиях? Приближает ли оно его к решению задачи?". Можно считать, что LLM-as-a-Judge является умным критиком, который видит весь контекст при решении задачи.

Оценку на каждом шаге можно записывать в словарь, который добавляется в `step_records`:
```
{
    "step_id": state.steps,
    "observation": observation_before,
    "agent_messages": copy.deepcopy(messages),
    "admissible_actions": list(admissible_actions),
    "raw_agent_output": raw_output,
    "parsed_action": parsed_action,
    "next_observation": observation_after,
    "reward": state.reward,

--->"actor_score": 1,  # Как пример, пока считаем, что скор бинарный - 0/1

    "done": state.finished,
}
```
Затем в скрипте с подсчетом метрик (`/scripts/calc_results_alfworld.py`) надо добавить расчет финальной оценка актора.
