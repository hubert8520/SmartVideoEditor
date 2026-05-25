# Editor Review Reports

Ten brief jest ostatnia warstwa pipeline'u, kiedy automatyczna naprawa nie
powinna juz zgadywac. Ma pokazac montazyscie ten sam problem w czasie finalnym i
w raw, razem z powodem decyzji QA.

## Wejscia

`scripts/generate_editor_review.py` czyta:

- `artifacts/final_quality_report*.json`,
- `artifacts/edit_decisions*.json`,
- `edited/edited_video*.mp4`,
- raw video z pola `source_video` w edit decisions.

Raport QA po aktualnym etapie powinien zawierac:

- `raw_ranges` - mapowanie problemu finalnego filmu na surowe nagranie,
- `raw_context` - slowa z raw wokol problemu,
- `repair_suggestion.action` - `force_keep`, `force_drop`, `manual_review` albo `no_auto_repair`,
- `actionability` - czy problem jest zmapowany i czy wymaga review.

Jesli QA report pochodzi ze starszej wersji i nie ma `raw_ranges`, brief uzywa
fallbacku przez `edit_decisions.timeline_map`.

## Wyniki

Skrypt zapisuje:

- `artifacts/editor_review*.md` - czytelny brief dla czlowieka,
- `artifacts/editor_review*.csv` - arkusz do sortowania i checklisty,
- opcjonalnie `artifacts/editor_review*_clips/`, gdy uzyjesz `--make-clips`.

## Pola w briefie

- `Akcja QA` mowi, co LLM uznal za najlepsza droge naprawy.
- `Status naprawy` mowi, czy to kandydat do automatycznej naprawy, czy reczny
  review.
- `Pewnosc QA` pomaga sortowac ryzyko.
- `Film edytowany` to dokladny zakres problemu w finalnym renderze.
- `Raw do porownania` to odpowiadajacy zakres w oryginalnym nagraniu, z
  marginesem do odsluchu.
- `Kontekst raw` pokazuje slowa z surowego transcriptu wokol problemu.
- `Instrukcja dla montazysty` tlumaczy, czy sprawdzic brakujacy lacznik,
  powtorke/noise, czy po prostu odsluchac przed decyzja.

## Zasada bezpieczenstwa

Brief nie wykonuje montazu. Pokazuje tylko, dlaczego QA albo repair planner
uwaza miejsce za bezpieczne do naprawy lub wymagajace review. Jesli decyzja jest
niepewna, wynik powinien zostac przy `manual_review`, a nie wymuszac agresywne
ciecie.

