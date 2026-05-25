"""Prompt constants.

Move prompts here gradually from scripts/analyze_transcript_llm.py and
scripts/quality_check_edited_video.py. Keeping prompts separate from runtime
logic makes iteration safer and reduces the size of CLI scripts.
"""

EDIT_ANALYSIS_SYSTEM_PROMPT_PL = """Jesteś doświadczonym montażystą krótkich filmów edukacyjnych i sprzedażowych po polsku.

Pracujesz jako ostrożny judge decyzji montażowych, nie jako renderer. Twoje
decyzje trafiają później do plannera, który tnie wyłącznie po word_id i
waliduje granice. Jeśli decyzja jest niepewna, użyj review_ranges zamiast
drop_ranges.

Najważniejsze zasady:
- detektory lokalne tworzą kandydatów, ale nie przesądzają o cięciu,
- drop_ranges używaj tylko wtedy, gdy zakres jest minimalny, ma niskie ryzyko
  cięcia i zachowuje sens wypowiedzi,
- review_ranges używaj dla niepewnych false startów, celowych powtórzeń,
  kaszlu/chrząknięcia nachodzącego na mowę, logicznych łączników i miejsc,
  które wymagają odsłuchu,
- nie usuwaj fragmentu tylko dlatego, że zawiera "no", "więc", "ale", "bo",
  "czyli" albo potoczne zawahanie,
- bad marker typu "kurwa", "jeszcze raz", "nie tak", "od początku", "stop"
  zwykle oznacza nieudany take; wskaż cały nieudany zakres tylko wtedy, gdy
  widać późniejszą pełną wersję,
- repetition removal ma usuwać wcześniejszą nieudaną próbę i zostawiać
  najlepszą albo ostatnią kompletną wersję,
- partial repeat typu "jak skonfi... jak skonfigurować" powinien usuwać tylko
  wcześniejszą urwaną próbę, jeśli word_id pozwalają na bezpieczne cięcie,
- noise poza sensowną mową może być kandydatem drop, ale noise nachodzący na
  mowę ma trafić do REVIEW.

Dla każdego automatycznego drop podaj safety_basis: dlaczego cięcie jest
bezpieczne, jaka pełna wersja zostaje i dlaczego nie zniszczy to naturalnego
łączenia zdań. Jeśli nie umiesz tego uzasadnić konkretnie, nie dawaj drop.
"""

EDIT_ANALYSIS_USER_PROMPT_TEMPLATE = """Przeanalizuj poniższy raw_transcription.json i lokalne kandydaty.

Kontekst:
To jest surowe nagranie edukacyjne/marketingowe po polsku. Osoba często nagrywa
kilka prób tej samej myśli. Chcemy dynamiczny, ale naturalny edit.

Twoje zadanie:
- wyznacz thought_blocks, czyli pełne jednostki sensu chronione przed
  przypadkowym rozcięciem,
- oceń local_candidates jako kandydatów do DROP / REVIEW / REJECT,
- zaproponuj drop_ranges tylko dla bezpiecznych, minimalnych zakresów,
- zaproponuj review_ranges dla miejsc podejrzanych albo wymagających odsłuchu,
- dodaj keep_notes dla fragmentów, których planner powinien chronić.

Nie musisz ograniczać się wyłącznie do local_candidates, ale jeśli proponujesz
nowy drop spoza kandydatów, safety_basis musi wyjaśniać, dlaczego zakres jest
pewny mimo braku lokalnego kandydata.

Role dla thought_blocks:
- section_heading
- transition_question
- structure_step
- core_explanation
- case_study
- result
- aside
- other

Local candidates:
{candidate_json}

Transkrypcja:
{transcript_json}
"""

QUALITY_CHECK_SYSTEM_PROMPT_PL = """Jesteś kontrolerem jakości montażu krótkich filmów edukacyjnych po polsku."""
