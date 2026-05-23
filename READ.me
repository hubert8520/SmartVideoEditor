# SmartVideoEditor

Pipeline do automatycznej transkrypcji, analizy LLM, montażu i kontroli jakości surowego filmu.

## 1. Przygotowanie środowiska

Używaj lokalnego venv:

```bash
./venv/bin/python -m pip install -r requirements.txt
```

Projekt korzysta z `imageio-ffmpeg`, więc nie musisz osobno instalować `ffmpeg` przez Homebrew.

## 2. Klucze API

Sekrety trzymaj w pliku `.env`. Ten plik jest ignorowany przez Git.

Przykład:

```bash
OPENAI_API_KEY=sk-...
DEEPGRAM_API_KEY=...
```

`DEEPGRAM_API_KEY` jest domyślnie używany do transkrypcji, bo Deepgram daje słowa z dokładnymi timestampami.

`OPENAI_API_KEY` jest używany do:

- analizy transkrypcji przez LLM,
- opcjonalnej transkrypcji przez OpenAI,
- kontroli jakości finalnego filmu po renderze.

## 3. Plik wejściowy

Wrzuć surowy film do folderu:

```text
raw/
```

Jeśli w folderze `raw/` jest tylko jeden plik wideo/audio, skrypty wybiorą go automatycznie.

Jeśli jest kilka plików, podawaj nazwę:

```bash
--video-name nazwa_pliku.mp4
```

## 4. Transkrypcja word-level

Domyślnie używaj Deepgram:

```bash
./venv/bin/python scripts/transcribe_video.py --language pl
```

To jest równoważne:

```bash
./venv/bin/python scripts/transcribe_video.py --provider deepgram --language pl
```

Domyślny model Deepgram:

```text
nova-3
```

Z rozpoznawaniem mówców:

```bash
./venv/bin/python scripts/transcribe_video.py --language pl --diarize
```

OpenAI jest nadal dostępne:

```bash
./venv/bin/python scripts/transcribe_video.py --provider openai --language pl
```

Domyślny model OpenAI:

```text
whisper-1
```

`gpt-4o-transcribe` bez diarizacji zwraca zbyt grube segmenty do precyzyjnego montażu, więc skrypt blokuje ten wariant, chyba że jawnie dodasz `--allow-coarse-openai`.

Wynik zapisuje się tutaj:

```text
artifacts/raw_transcription.json
```

Nowy format ma segmenty i słowa:

```json
{
  "version": "2.0",
  "source": {
    "provider": "deepgram",
    "model": "nova-3",
    "word_level": true
  },
  "segments": [
    {
      "id": 0,
      "timestamp": "00:00:01:040",
      "end": "00:00:05:600",
      "transcription": "Tekst wypowiedzi",
      "word_ids": [0, 1, 2]
    }
  ],
  "words": [
    {
      "id": 0,
      "timestamp": "00:00:01:040",
      "end": "00:00:01:260",
      "word": "Tekst"
    }
  ]
}
```

Stary format listy segmentów nadal jest czytany jako fallback, ale wtedy `word_id` są tylko syntetyczne. Do dobrego montażu zrób ponowną transkrypcję Deepgramem.

## 5. Analiza transkrypcji przez LLM

Ten krok używa OpenAI i tworzy decyzje montażowe na podstawie sensu wypowiedzi oraz `word_id`.

Test bez użycia API:

```bash
./venv/bin/python scripts/analyze_transcript_llm.py --dry-run
```

Właściwa analiza:

```bash
./venv/bin/python scripts/analyze_transcript_llm.py
```

Domyślny model:

```text
gpt-5.2
```

Wynik:

```text
artifacts/llm_edit_decisions.json
```

Ten plik zawiera m.in.:

- `thought_blocks` - pełne jednostki sensu, których planner nie powinien przypadkowo rozcinać,
- `drop_ranges` - fragmenty do wycięcia, z `start_word_id` i `end_word_id`,
- `review_ranges` - fragmenty do ręcznego sprawdzenia,
- `keep_notes` - fragmenty, których raczej nie należy usuwać,
- `overall_notes` - ogólne uwagi o nagraniu.

## 6. Montaż filmu

```bash
./venv/bin/python scripts/edit_video.py --padding 0.05
```

`--padding` oznacza, ile sekund zostawić przed i po zachowanej kwestii.

Edytor:

- automatycznie używa `artifacts/llm_edit_decisions.json`, jeśli plik istnieje,
- traktuje lokalne heurystyki jako review, a nie jako automatyczne cięcia, chyba że dodasz `--allow-heuristic-drops`,
- chroni `thought_blocks`, `keep_notes`, nagłówki sekcji i podejrzane sklejenia,
- planuje cięcia po granicach słów,
- buduje keepy z maski słów, więc kaszel/odchrząknięcie/szuranie bez rozpoznanych słów zwykle wypada z filmu,
- używa krótszego ogona dla krótkich bloków typu `Punkt pierwszy`,
- przy krótkich nagłówkach i elementach struktury szuka też bardzo krótkiej ciszy po ostatnim słowie i przycina ogon do tej ciszy, żeby usuwać odchrząknięcia po frazie,
- dodaje mały margines bezpieczeństwa wokół wycinanych słów, ale nie pozwala mu wejść w sąsiednie słowo,
- snapuje cięcia do pobliskiej ciszy tylko wtedy, gdy nie narusza to granic słów,
- domyślnie buduje film z zakresów transkrypcji, więc dźwięki bez tekstu nie są zachowywane tylko dlatego, że ffmpeg wykrył tam "nie-ciszę",
- po renderze próbuje uruchomić kontrolę jakości finalnego filmu.

Dry-run bez renderowania:

```bash
./venv/bin/python scripts/edit_video.py --padding 0.05 --dry-run
```

Przydatne opcje:

```bash
# Zignoruj decyzje LLM
./venv/bin/python scripts/edit_video.py --ignore-llm-decisions

# Zmień minimalną pewność automatycznych cięć LLM
./venv/bin/python scripts/edit_video.py --llm-min-confidence 0.85

# Zmień margines wokół słów wycinanych przez planner
./venv/bin/python scripts/edit_video.py --cut-safety-margin 0.08

# Dostosuj maskę słów i ogon po krótkich blokach
./venv/bin/python scripts/edit_video.py --word-head-padding 0.05 --word-tail-padding 0.06 --short-block-tail-padding 0.02

# Dostosuj przycinanie ogona krótkich nagłówków/elementów struktury do krótkiej ciszy
./venv/bin/python scripts/edit_video.py --short-block-silence-min-duration 0.08 --short-block-silence-window 0.45 --short-block-min-spoken-before-trim 0.25

# Wyłącz przycinanie krótkich nagłówków/elementów struktury do krótkiej ciszy
./venv/bin/python scripts/edit_video.py --disable-short-block-silence-trim

# Wyłącz maskę słów i wróć do paddingu całych interwałów
./venv/bin/python scripts/edit_video.py --disable-word-mask

# Pozwól starym heurystykom robić automatyczne cięcia
./venv/bin/python scripts/edit_video.py --allow-heuristic-drops

# Wyłącz walidator granic myśli
./venv/bin/python scripts/edit_video.py --disable-boundary-validator

# Wróć do starego trybu opartego głównie o nie-ciszę z audio
./venv/bin/python scripts/edit_video.py --keep-source audio

# Nie uruchamiaj kontroli jakości po renderze
./venv/bin/python scripts/edit_video.py --skip-quality-check
```

Wyniki:

```text
artifacts/edit_decisions.json
edited/edited_video.mp4
```

Każdy `edit_decisions*.json` zawiera też `timeline_map`, czyli mapę czasu finalnego filmu na czas w surowym nagraniu. Dzięki temu kolejne poprawki zawsze renderują od nowa z pliku w `raw/`, a nie z już pociętego filmu.

## 7. Kontrola jakości finalnego filmu

`edit_video.py` próbuje uruchomić ten krok automatycznie po renderze. Możesz też odpalić go ręcznie:

```bash
./venv/bin/python scripts/quality_check_edited_video.py --language pl
```

Ten skrypt:

- transkrybuje `edited/edited_video.mp4`, domyślnie Deepgramem,
- zapisuje `artifacts/edited_transcription.json`,
- wysyła finalną transkrypcję do LLM,
- zapisuje raport w `artifacts/final_quality_report.json`.

Raport wykrywa m.in. urwane słowa, niepotrzebne powtórzenia, zawieszone myśli i nielogiczne przejścia.

## 8. Iteracyjna naprawa po QA

Jeśli `final_quality_report.json` ma status `fail` albo `needs_review`, wygeneruj plan napraw:

```bash
./venv/bin/python scripts/repair_from_quality_report.py
```

Wynik:

```text
artifacts/repair_plan.json
```

Potem wyrenderuj drugą wersję z oryginalnego filmu:

```bash
./venv/bin/python scripts/edit_video.py \
  --padding 0.1 \
  --repair-plan artifacts/repair_plan.json \
  --output edited/edited_video_v2.mp4 \
  --edit-decisions-output artifacts/edit_decisions_v2.json \
  --quality-language pl \
  --quality-transcript-output artifacts/edited_transcription_v2.json \
  --quality-report-output artifacts/final_quality_report_v2.json
```

Możesz też puścić ograniczoną pętlę automatyczną. Domyślnie robi maksymalnie v2 i v3, a potem zatrzymuje się do ręcznego review, jeśli QA nadal nie przejdzie:

```bash
./venv/bin/python scripts/auto_refine_video.py --quality-language pl
```

Naprawy są konserwatywne:

- `force_drop_words` - wymusza usunięcie powtórzeń potwierdzonych przez QA,
- `force_keep_interval` - przywraca krótki fragment audio między słowami, gdy QA wykryje brakujący łącznik,
- `force_keep_words` - poszerza keep dla słowa rozpoznanego w raw, ale zgubionego w finalnej transkrypcji.

## 9. Brief dla montażysty

JSON-y są dla skryptów. Dla osoby montującej wygeneruj prosty brief:

```bash
./venv/bin/python scripts/generate_editor_review.py --make-clips
```

Domyślnie skrypt wybiera najnowszy `final_quality_report_v*.json` i odpowiadający mu `edit_decisions_v*.json`.

Wyniki:

```text
artifacts/editor_review_v3.md
artifacts/editor_review_v3.csv
artifacts/editor_review_v3_clips/
```

Brief zawiera:

- czas problemu w filmie edytowanym,
- odpowiadający czas w raw,
- opis co brzmi podejrzanie,
- sugestię co sprawdzić/poprawić,
- krótkie klipy porównawcze edit/raw, jeśli użyjesz `--make-clips`.

## 10. Typowy workflow

```bash
# 1. Wrzuć film do raw/

# 2. Zrób transkrypcję word-level
./venv/bin/python scripts/transcribe_video.py --language pl

# 3. Przeanalizuj transkrypcję przez LLM
./venv/bin/python scripts/analyze_transcript_llm.py

# 4. Zmontuj film i uruchom QA po renderze
./venv/bin/python scripts/edit_video.py --padding 0.05 --quality-language pl

# 5. Jeśli QA zwróci fail/needs_review, uruchom ograniczoną pętlę naprawczą
./venv/bin/python scripts/auto_refine_video.py --quality-language pl

# 6. Jeśli po automatyce nadal jest needs_review, przygotuj brief dla montażysty
./venv/bin/python scripts/generate_editor_review.py --make-clips
```

Wynikowy film znajdziesz tutaj:

```text
edited/edited_video.mp4
```

## 11. Pliki lokalne ignorowane przez Git

Nie commituj:

```text
.env
raw/
artifacts/
edited/
venv/
```

Te ścieżki są w `.gitignore`.
