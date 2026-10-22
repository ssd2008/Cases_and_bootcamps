# Cases_and_bootcamps

Архив кейсов, олимпиад и буткемпов. Каждый проект находится в собственной ветке.

`main` — каталог. Для кода откройте нужную ветку по ссылке в таблице. Ветки содержат полную исходную историю; истории разных проектов не объединены.

| Источник / код | Проект | Ветка | Презентация |
|---|---|---|---|
| [HseCase](https://github.com/ssd2008/Cases_and_bootcamps/tree/codex/HseCase) | Кейс ВШЭ: анализ здоровья и роли спорта | `codex/HseCase` | **Нужно добавить** |
| [kodiim25spring](https://github.com/ssd2008/Cases_and_bootcamps/tree/codex/kodiim25spring) | КодИИМ, весна 2025: ИИ-ассистент X5 | `codex/kodiim25spring` | **Нужно добавить** |
| [GMW](https://github.com/ssd2008/Cases_and_bootcamps/tree/codex/GMW) | Большая Математическая мастерская | `codex/GMW` | **Нужно добавить, если были слайды** |
| [Reshis](https://github.com/ssd2008/Cases_and_bootcamps/tree/codex/Reshis) | Олимпиада «Решись» Центрального университета | `codex/Reshis` | **Нужно добавить, если были слайды** |
| [ReshisRazrab](https://github.com/ssd2008/Cases_and_bootcamps/tree/codex/ReshisRazrab) | Олимпиада «Решись»: дополнительная часть | `codex/ReshisRazrab` | **Нужно добавить, если были слайды** |
| [PRODmsk](https://github.com/ssd2008/Cases_and_bootcamps/tree/codex/PRODmsk) | Олимпиадный проект PRODmsk: Swift-приложение | `codex/PRODmsk` | **Нужно добавить, если были слайды** |
| [YaLyceum25](https://github.com/ssd2008/Cases_and_bootcamps/tree/codex/YaLyceum25) | Яндекс Лицей: проект специализации, 2025 | `codex/YaLyceum25` | Есть — PDF |

## Работа локально

```bash
git clone git@github.com:ssd2008/Cases_and_bootcamps.git
cd Cases_and_bootcamps
git branch -r
git switch codex/HseCase
```

Для нескольких проектов одновременно используйте `git worktree`. Перед переключением сохраните локальные изменения.

## Что сохранено

- Исходные файлы и полная история коммитов, авторы и даты.
- Все исходные ветки; дополнительные ветки имеют суффикс `--<исходная-ветка>`.
- Исходные теги, если они были, — в пространстве имён `<источник>/<тег>`.
- Происхождение и SHA коммитов в [migration.json](migration.json); детали проекта — в `ARCHIVE.md` его ветки.

Issues, PR, настройки доступа и секреты Actions принадлежат старым репозиториям и не переносятся вместе с Git. Все исходные репозитории сохранены как архивы; YaLyceum25 переведён в Private Archive. GMW, Reshis, ReshisRazrab и PRODmsk переведены в Private Archive по указанию владельца; их код остаётся доступным в ветках этой публичной коллекции. Названия событий основаны на исходных репозиториях; места и награды не добавлялись.

Собрано 2026-10-22. Обновлено 2026-10-23.

`pulse` сохранён отдельным приватным архивом: это fork приватного репозитория Центрального университета. Его код в публичную коллекцию не включён.
