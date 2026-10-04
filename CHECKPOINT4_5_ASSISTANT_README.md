# OncoTwin CP4.5 — Ask OncoTwin

This increment adds a patient-facing conversational interpreter on top of the verified longitudinal state and frozen trajectory model.

## Demo mode

When Maya fixture mode is enabled, the assistant performs no provider calls. Suggested questions are broad, while a larger library of exact authored questions can be typed into the same free-form chat box. Authored answers and follow-ups are deterministic. Forecast values shown inside selected responses are injected from the real frozen forecast rather than hardcoded. Unsupported questions fail closed and redirect to suggested questions.

## Normal mode

The same API and UI call the existing CP4 shared LLM router with the user's free-form question, recent conversation turns, verified patient state, verified facts/source keys, current frozen forecast context, and any already-retrieved evidence bundle. The assistant is instructed to keep record facts, model estimates, and evidence context separate; it may not recommend treatment, assert trial eligibility, invent sources, or attribute a forecast shift to an individual lesion.

## UI

`AskOncoTwin` is embedded on the Overview and after the newest-scan update flow. Broad suggested questions guide the experience; a standard text box supports typed questions and follow-up questions.
