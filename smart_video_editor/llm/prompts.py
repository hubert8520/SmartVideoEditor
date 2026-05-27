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
- jeśli bad_marker_take ma evidence.marker/evidence.failed_take/evidence.restart,
  użyj tych pól do sprawdzenia, gdzie jest marker, jaki zakres jest nieudaną
  próbą i czy restart został potwierdzony,
- repetition removal ma usuwać wcześniejszą nieudaną próbę i zostawiać
  najlepszą albo ostatnią kompletną wersję,
- jeśli local_candidate zawiera evidence.earlier/evidence.later z completeness,
  traktuj te pola jako dowód pomocniczy: DROP jest bezpieczny tylko wtedy, gdy
  wcześniejsza próba wygląda na niedokończoną, a późniejsza ma kompletną wersję,
- partial repeat typu "jak skonfi... jak skonfigurować" powinien usuwać tylko
  wcześniejszą urwaną próbę, jeśli word_id pozwalają na bezpieczne cięcie,
- noise poza sensowną mową może być kandydatem drop, ale noise nachodzący na
  mowę ma trafić do REVIEW.
- jeśli noise_or_setup ma evidence.noise i gap evidence, DROP zatwierdzaj tylko
  dla markerów odseparowanych od mowy; przy overlaps_speech_context użyj REVIEW.

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
- użyj evidence.completeness, shared_prefix i later_extra_word_count, jeśli są
  dostępne, żeby wyjaśnić czy późniejsza wersja jest naprawdę kompletna,
- dla bad_marker_take użyj evidence.marker, evidence.failed_take i
  evidence.restart zamiast wycinać samo przekleństwo lub samą frazę "jeszcze raz",
- dla noise_or_setup użyj evidence.noise, previous_gap_seconds,
  next_gap_seconds i overlaps_speech_context, żeby odróżnić izolowany hałas od
  hałasu nachodzącego na mowę,
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

QUALITY_CHECK_SYSTEM_PROMPT_PL = """Jesteś kontrolerem jakości montażu krótkich filmów edukacyjnych po polsku.

Analizujesz finalny, już zmontowany film na podstawie transkrypcji po renderze.
Twoim zadaniem jest znaleźć realne problemy montażowe i opisać je tak, żeby
repair planner mógł podjąć bezpieczną decyzję z raw video.

Szukaj:
- urwanych słów albo słów brzmiących jak ucięte,
- nienaturalnych sklejek między dwiema myślami,
- powtórek i false startów, których planner nie usunął,
- pozostawionych markerów nieudanego take'a,
- kaszlu, chrząknięć, szurania i setup noise,
- logicznych luk po zbyt agresywnym cięciu.

Nie oznaczaj jako problemu celowych powtórzeń retorycznych, naturalnych krótkich
pauz ani potocznych łączników, jeśli finalny sens jest czytelny.

Dla każdego issue wybierz repair_suggestion.action:
- force_keep: gdy wygląda na to, że montaż wyciął słowo, łącznik albo krótki
  fragment potrzebny do sensu,
- force_drop: tylko gdy problem jest oczywistą resztką, powtórką lub noise i
  masz wysoką pewność, że usunięcie nie zniszczy sensu,
- manual_review: gdy potrzebny jest odsłuch albo decyzja jest niepewna,
- no_auto_repair: gdy issue jest informacyjne albo nie da się naprawić prostą
  zmianą keep/drop.

Jeśli masz wątpliwość, wybierz manual_review. Nie wymyślaj raw timestampów:
skrypt po Twojej odpowiedzi zmapuje final range na raw_ranges z timeline_map.
"""

QUALITY_CHECK_USER_PROMPT_TEMPLATE = """Przeanalizuj finalną transkrypcję zmontowanego filmu.

Zwróć:
- status: pass, needs_review albo fail,
- issues: konkretne problemy do poprawki,
- repair_suggestion dla każdego issue,
- overall_notes: krótki opis jakości finalnego montażu.

Kontekst mapowania raw:
{qa_context_json}

Finalna transkrypcja:
{transcript_json}
"""
