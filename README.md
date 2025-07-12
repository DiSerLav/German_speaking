# Итоговый отчёт по Выпускной Квалификационной Работе

| № | Раздел                            | Файл                                                          |
|:-:|:----------------------------------|:--------------------------------------------------------------|
| 0 | Титульная страница                | [00_title.md](final_work/00_title.md)                         |
| 1 | Введение                          | [01_introduction.md](final_work/01_introduction.md)           |
| 2 | Обзор литературы                  | [02_literature_review.md](final_work/02_literature_review.md) |
| 3 | Методология исследования          | [03_methodology.md](final_work/03_methodology.md)             |
| 4 | Реализация программного комплекса | [04_implementation.md](final_work/04_implementation.md)       |
| 5 | Результаты и обсуждение           | [05_results.md](final_work/05_results.md)                     |
| 6 | Заключение                        | [06_conclusion.md](final_work/06_conclusion.md)               |
| 7 | Список источников                 | [07_bibliography.md](final_work/07_bibliography.md)           |

# Запуск

## Подготовка компьютера

1. Установить python 3.13
2. Выполнить команды 

```bash
pip install poetry
cd /путь/к/проекту
poetry install
```

## Запуск пайплайна

Запуск выполняется из корневой директории проекта (где есть файл [pyproject.toml](pyproject.toml))

```bash
poetry run python -m ai_lab.pipline
```

## Посмотреть примеры из каждого топика

```bash
poetry run python -m ai_lab.show_samples
```

## Запуск интерфейса

```bash
poetry run python -m ai_lab.pipline
```
