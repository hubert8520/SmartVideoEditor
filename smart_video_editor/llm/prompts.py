"""Prompt constants.

Move prompts here gradually from scripts/analyze_transcript_llm.py and
scripts/quality_check_edited_video.py. Keeping prompts separate from runtime
logic makes iteration safer and reduces the size of CLI scripts.
"""

EDIT_ANALYSIS_SYSTEM_PROMPT_PL = """Jesteś doświadczonym montażystą krótkich filmów edukacyjnych i sprzedażowych po polsku."""

QUALITY_CHECK_SYSTEM_PROMPT_PL = """Jesteś kontrolerem jakości montażu krótkich filmów edukacyjnych po polsku."""
