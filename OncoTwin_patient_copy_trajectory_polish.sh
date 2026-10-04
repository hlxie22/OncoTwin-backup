#!/usr/bin/env bash
set +e
(
  set -euo pipefail

  ROOT="${1:-/home/henryxie/OncoTwin-backup}"
  cd "$ROOT"

  if [ -f scripts/app/common_env.sh ]; then
    source scripts/app/common_env.sh
  fi

  TS="$(date +%Y%m%d_%H%M%S)"
  ART="artifacts/app_patient_copy_polish_${TS}"
  BACKUP="$ART/backup"
  mkdir -p "$BACKUP"

  FILES=(
    apps/oncotwin-web/components/AppShell.tsx
    apps/oncotwin-web/app/app/page.tsx
    apps/oncotwin-web/app/app/forecast/page.tsx
    apps/oncotwin-web/app/app/update/page.tsx
    apps/oncotwin-web/app/app/timeline/page.tsx
    apps/oncotwin-web/app/app/records/page.tsx
    apps/oncotwin-web/app/app/verify/page.tsx
    apps/oncotwin-web/components/ModelSupport.tsx
    apps/oncotwin-web/components/ForecastChart.tsx
  )

  echo "=== BACKING UP TARGET FILES ==="
  for f in "${FILES[@]}"; do
    if [ ! -f "$f" ]; then
      echo "Missing required file: $f" >&2
      exit 1
    fi
    mkdir -p "$BACKUP/$(dirname "$f")"
    cp -p "$f" "$BACKUP/$f"
    echo "backup: $f"
  done

  echo
  echo "=== APPLYING PATIENT-LANGUAGE + NAVIGATION POLISH ==="

  python - <<'PY'
from pathlib import Path
import re

changed = []

def load(path: str):
    p = Path(path)
    return p, p.read_text()

def save(p: Path, old: str, new: str):
    if new != old:
        p.write_text(new)
        changed.append(str(p))

def repl(s: str, old: str, new: str, label: str, *, required=True):
    if old in s:
        return s.replace(old, new)
    if new in s:
        return s
    if required:
        raise RuntimeError(f"Could not find expected text for {label}: {old!r}")
    return s

def regex_repl(s: str, pattern: str, replacement: str, label: str, *, required=True, flags=0):
    out, n = re.subn(pattern, replacement, s, flags=flags)
    if n:
        return out
    # idempotency: if replacement is a literal-ish string and already present, accept it.
    if replacement and replacement in s:
        return s
    if required:
        raise RuntimeError(f"Could not match expected pattern for {label}: {pattern!r}")
    return s

# -----------------------------------------------------------------------------
# 1) App shell: one patient-facing Trajectory destination; scan upload is an
#    action inside that experience, not a separate navigation tab.
#    Keep the hosted-build footer to the single requested label.
# -----------------------------------------------------------------------------
p, s = load("apps/oncotwin-web/components/AppShell.tsx")
old = s
s = regex_repl(
    s,
    r"\n\s*\['/app/update'\s*,\s*'[^']+'\]\s*,?",
    "",
    "remove Add Newest Scan nav tab",
    required=False,
)
s = regex_repl(
    s,
    r"\s*<span>\s*Independent external confirmation is pending\.\s*</span>",
    "",
    "remove repeated external-confirmation footer copy",
    required=False,
)
s = regex_repl(
    s,
    r"\s*<span>\s*Use synthetic or de-identified records in this hosted build\.\s*</span>",
    "",
    "remove hosted-build synthetic-record footer copy",
    required=False,
)
save(p, old, s)

# -----------------------------------------------------------------------------
# 2) Overview: plain-language title, no state hash, lighter forecast language,
#    no 'Missing != negative' callout.
# -----------------------------------------------------------------------------
p, s = load("apps/oncotwin-web/app/app/page.tsx")
old = s
s = repl(s, "<h1>What OncoTwin knows right now</h1>", "<h1>Your cancer right now</h1>", "overview title")
s = regex_repl(
    s,
    r"<p>\{patient\?\.display_name\}\. Important facts stay connected to the reports they came from\.</p>",
    "<p>A summary of what your records show today.</p>",
    "overview subtitle",
)
s = regex_repl(
    s,
    r"\s*<span className=\"state-hash\">STATE \{data\.state_hash\?\.slice\(0, 8\)\.toUpperCase\(\) \|\| 'EMPTY'\}</span>",
    "",
    "remove overview state hash",
)
s = repl(
    s,
    "<p>OncoTwin will keep the before-scan model state separate, add only the new scan evidence, and show what changed.</p>",
    "<p>Add a new scan when you get one. OncoTwin will show what changed and update your trajectory.</p>",
    "overview newest-scan copy",
)
s = repl(s, '<div><div className="eyebrow">RESEARCH TRAJECTORY</div><h2>Your latest model estimate</h2></div>', '<div><div className="eyebrow">TRAJECTORY</div><h2>Your latest outlook</h2></div>', "overview trajectory heading")
s = repl(
    s,
    "Once your record includes a dated metastatic-disease anchor and a verified scan, OncoTwin can run the frozen research model.",
    "Once your records include when metastatic cancer was diagnosed and at least one scan, OncoTwin can show your trajectory.",
    "overview forecast empty copy",
)
s = repl(
    s,
    "'Progression-free survival research estimate'",
    "'Your latest progression-free outlook'",
    "overview forecast title",
)
s = repl(
    s,
    "'These numbers come from the frozen V2-07 model and are not a treatment recommendation.'",
    "'See your latest trajectory and how it changed after your newest scan.'",
    "overview forecast description",
)
s = repl(
    s,
    '<div className="section-title"><div><div className="eyebrow">DATA COMPLETENESS</div><h2>What is still missing</h2></div><span className="pill">Missing ≠ negative</span></div>',
    '<div className="section-title"><div><div className="eyebrow">YOUR RECORDS</div><h2>What is still missing</h2></div></div>',
    "remove missing-negative pill",
)
s = repl(s, "No reliable committed fact is available yet.", "This information has not been confirmed from your records yet.", "missing-info copy")
s = repl(s, "Core patient-state fields are present.", "Your core cancer information is available.", "complete-state copy")
s = repl(s, "That does not mean every clinically useful record has been uploaded.", "You can keep adding records as your care changes.", "complete-state subcopy")
save(p, old, s)

# -----------------------------------------------------------------------------
# 3) Trajectory page: keep the science untouched, but translate the main screen
#    into patient language. Technical mechanics remain behind the disclosure.
# -----------------------------------------------------------------------------
p, s = load("apps/oncotwin-web/app/app/forecast/page.tsx")
old = s
s = repl(s, 'aria-label="Loading research trajectory"', 'aria-label="Loading your trajectory"', "trajectory loading label")
s = repl(s, '<div className="eyebrow">RESEARCH TRAJECTORY</div>', '<div className="eyebrow">TRAJECTORY</div>', "trajectory eyebrow")
s = repl(s, '<h1>How the estimate changes with new scan evidence</h1>', '<h1>How your outlook is changing</h1>', "trajectory title")
s = repl(
    s,
    '<p>This is a research-model estimate, not a treatment recommendation. Independent external confirmation is still pending.</p>',
    '<p>See how your outlook changes as new scans are added to your history.</p>',
    "trajectory header copy",
)
s = repl(s, '<p>Recalculate to align the trajectory with the current committed state.</p>', '<p>Recalculate to use your latest records.</p>', "stale forecast copy")
s = repl(s, '<h2>Check whether the frozen model can use your current record</h2>', '<h2>Build your trajectory from your current records</h2>', "empty trajectory title")
s = repl(
    s,
    '<p>The model only uses committed facts. It requires a known model clock and a verified current scan assessment; otherwise it will return an explicit unavailable state instead of guessing.</p>',
    '<p>OncoTwin needs a dated metastatic diagnosis and at least one scan before it can show a trajectory.</p>',
    "empty trajectory copy",
)
s = repl(s, "'Calculate research forecast'", "'Show my trajectory'", "trajectory calculate button")
s = repl(
    s,
    '<p>PRE is the model state before the newest scan. The selected update adds only that scan evidence using the frozen partial-update rule.</p>',
    '<p>Compare your outlook before the newest scan with your outlook after that scan was added.</p>',
    "main trajectory explanation",
)
s = repl(s, '<p>Selected PFS estimate</p>', '<p>After this scan</p>', "horizon label")
s = repl(s, '<span>Before scan {pct(h?.pre_pfs)}</span>', '<span>Before this scan {pct(h?.pre_pfs)}</span>', "horizon before label")
s = repl(
    s,
    '<div className="unavailable-panel"><strong>A quantitative curve is unavailable for this state.</strong><p>The limitation is shown below; OncoTwin does not fill missing model inputs with guesses.</p></div>',
    '<div className="unavailable-panel"><strong>OncoTwin cannot show a trajectory yet.</strong><p>See below for the information that is still needed.</p></div>',
    "unavailable trajectory copy",
)

# Replace the three explanatory cards as one block so visible PRE/POST/alpha jargon disappears.
pattern = r'''\s*<section className="explain-grid">.*?</section>\n\n\s*<section className="section-block support-section">'''
replacement = '''
          <section className="explain-grid">
            <article className="explain-card">
              <div className="explain-number">1</div>
              <div><div className="card-kicker">BEFORE THIS SCAN</div><h3>Your earlier outlook</h3><p>Based on the history in your record before the newest scan.</p></div>
            </article>
            <article className="explain-card">
              <div className="explain-number">2</div>
              <div><div className="card-kicker">AFTER THIS SCAN</div><h3>Your updated outlook</h3><p>Updated after adding the newest scan to that same history.</p></div>
            </article>
            <article className="explain-card">
              <div className="explain-number">3</div>
              <div><div className="card-kicker">WHAT CHANGED</div><h3>The effect of the new scan</h3><p>The difference shows how much the newest scan changed the estimate.</p></div>
            </article>
          </section>

          <section className="section-block support-section">'''
s = regex_repl(s, pattern, replacement, "simplify trajectory explanation cards", flags=re.S)
s = repl(s, '<div className="section-title"><div><div className="eyebrow">MODEL SUPPORT</div><h2>Where this estimate is limited</h2></div></div>', '<div className="section-title"><div><div className="eyebrow">ABOUT THIS ESTIMATE</div><h2>How much information is available</h2></div></div>', "support section heading")
s = repl(s, '<summary>Model details and full POST output</summary>', '<summary>Technical model details</summary>', "technical details summary")
s = repl(
    s,
    '<p>The patient-facing selected curve is the frozen deployment output. The full POST curve is shown only as technical context.</p>',
    '<p>These details are included for technical review and reproducibility.</p>',
    "technical details intro",
)
s = repl(s, "'Hide full POST curve'", "'Hide full technical curve'", "hide POST button")
s = repl(s, "'Show full POST curve'", "'Show full technical curve'", "show POST button")
s = regex_repl(
    s,
    r'''\s*<div><dt>Research status</dt><dd>Independent external confirmation pending</dd></div>''',
    "",
    "remove external-confirmation metadata",
    required=False,
)
save(p, old, s)

# -----------------------------------------------------------------------------
# 4) New-scan flow: this remains a route/action, but not a separate nav tab.
#    Rewrite the visible flow around the patient story, not model internals.
# -----------------------------------------------------------------------------
p, s = load("apps/oncotwin-web/app/app/update/page.tsx")
old = s
s = repl(s, "New genomic information in the committed record:", "New genomic information in your record:", "update genomics copy")
s = repl(s, "The committed patient state changed, but no headline summary field changed. The underlying source-linked history still advanced.", "Your cancer history was updated, even though the main summary did not change.", "update state fallback")
s = repl(s, "No additional headline patient-state field changed beyond the scan evidence used for this update.", "No other headline information changed with this scan.", "update no-change fallback")
s = repl(s, "Reading the report and connecting important facts to their source pages…", "Reading the report and connecting important details to the right source pages…", "scan processing message")
s = repl(s, "A few important scan facts need confirmation before they can change the patient state.", "A few important details need your confirmation before they are added to your cancer history.", "scan verification message")
s = repl(s, "Building the updated patient state and running the frozen research model…", "Updating your cancer history and trajectory…", "scan update running message")
s = repl(s, "Update complete. The before-scan model state stayed separate from the new scan evidence.", "Update complete. Your newest scan is now part of your cancer history and trajectory.", "scan complete message")
s = repl(s, '<div><div className="eyebrow">NEW SCAN UPDATE</div><h1>Add the newest scan without losing the before-scan picture</h1><p>OncoTwin keeps the history-only PRE model state separate, then adds only the verified current scan evidence for the selected update.</p></div>', '<div><div className="eyebrow">NEW SCAN</div><h1>Add your newest scan</h1><p>OncoTwin will add the scan to your history and show how your trajectory changes.</p></div>', "new scan header")
s = repl(s, '<div className="card-kicker">BEFORE NEW SCAN</div>', '<div className="card-kicker">BEFORE THIS SCAN</div>', "before scan kicker")
s = repl(s, "'The current state is snapshotted before the new report is processed.'", "'This is the latest scan already in your history.'", "before scan fallback")

before_metrics_pattern = r'''<div className="before-metrics">\s*<span>Current state</span><strong>\{beforeState\?\.state_hash\?\.slice\(0, 8\)\.toUpperCase\(\) \|\| 'EMPTY'\}</strong>\s*\{priorForecast\?\.horizons\?\.\['6'\] \? <><span>Latest 6m research estimate</span><strong>\{pct\(priorForecast\.horizons\['6'\]\.selected_pfs\)\}</strong></> : null\}\s*</div>'''
before_metrics_replacement = '''<div className="before-metrics">
            <span>Previous 6-month outlook</span>
            <strong>{priorForecast?.horizons?.['6'] ? pct(priorForecast.horizons['6'].selected_pfs) : 'Not calculated'}</strong>
          </div>'''
s = regex_repl(s, before_metrics_pattern, before_metrics_replacement, "remove state hash from scan flow", flags=re.S)
s = repl(s, "'The committed patient state is ready for the frozen PRE → selected scan update.'", "'Your scan is ready to be added to your history and trajectory.'", "ready update copy")
s = repl(s, "'Run scan update'", "'Update my trajectory'", "run update button")
s = regex_repl(
    s,
    r'''<section className="after-banner"><div><div className="card-kicker">AFTER NEW SCAN</div><h2>Your patient state has been updated</h2><p>State \{beforeState\?\.state_hash\?\.slice\(0, 8\)\.toUpperCase\(\)\} → \{afterState\.state_hash\?\.slice\(0, 8\)\.toUpperCase\(\)\}</p></div><ModelSupport support=\{forecast\.support\} compact /></section>''',
    '''<section className="after-banner"><div><div className="card-kicker">AFTER THIS SCAN</div><h2>Your cancer history has been updated</h2><p>Your newest scan is now part of your history and trajectory.</p></div><ModelSupport support={forecast.support} compact /></section>''',
    "remove state hashes from completed update",
)
s = repl(s, '<div><div className="card-kicker">HOW THE RESEARCH FORECAST CHANGED</div>', '<div><div className="card-kicker">HOW YOUR OUTLOOK CHANGED</div>', "update forecast card label")
s = repl(s, '<p>Before this scan: {pct(forecast.horizons[\'6\'].pre_pfs)} · Selected update: {pct(forecast.horizons[\'6\'].selected_pfs)}.</p>', '<p>Before this scan: {pct(forecast.horizons[\'6\'].pre_pfs)} · After this scan: {pct(forecast.horizons[\'6\'].selected_pfs)}.</p>', "update before-after copy")
s = repl(s, '<div className="card-kicker">SAME LANDMARK, BEFORE → AFTER</div><h2>Before this scan vs. selected update</h2><p>The selected curve uses the frozen α = {forecast.model?.locked_alpha ?? 0.52} partial PRE-to-POST update.</p>', '<div className="card-kicker">BEFORE → AFTER</div><h2>How this scan changed your trajectory</h2><p>Compare your outlook before this scan with your updated outlook after the scan was added.</p>', "update chart heading")
s = repl(s, '<p>Selected PFS estimate</p>', '<p>After this scan</p>', "update horizon label")
s = repl(s, '<section className="section-block support-section"><div className="section-title"><div><div className="eyebrow">MODEL SUPPORT</div><h2>What to keep in mind</h2></div></div><ModelSupport support={forecast.support} /></section>', '<section className="section-block support-section"><div className="section-title"><div><div className="eyebrow">ABOUT THIS ESTIMATE</div><h2>How much information is available</h2></div></div><ModelSupport support={forecast.support} /></section>', "update support heading")
s = repl(s, '<Link href="/app/forecast" className="primary-button">Open full trajectory</Link>', '<Link href="/app/forecast" className="primary-button">Back to trajectory</Link>', "update return button")
s = repl(s, 'Start another scan update', 'Add another scan', "another scan button")
save(p, old, s)

# -----------------------------------------------------------------------------
# 5) Timeline, records, and review: remove explanatory/jargony copy that does
#    not help a patient complete the task.
# -----------------------------------------------------------------------------
p, s = load("apps/oncotwin-web/app/app/timeline/page.tsx")
old = s
s = repl(
    s,
    '<header className="page-head"><div><div className="eyebrow">YOUR CANCER HISTORY</div><h1>One timeline, built from your sources</h1><p>Pathology, treatment, scans, disease-site evidence, and genomics appear as dated events rather than isolated chat messages.</p></div></header>',
    '<header className="page-head"><div><div className="eyebrow">YOUR CANCER HISTORY</div><h1>Your cancer timeline</h1></div></header>',
    "timeline header",
)
save(p, old, s)

p, s = load("apps/oncotwin-web/app/app/records/page.tsx")
old = s
s = repl(s, '<div><div className="eyebrow">YOUR RECORDS</div><h1>One record library, one cancer history</h1><p>Pathology, radiology, genomics, treatment summaries, labs, and portal exports stay connected to the facts OncoTwin extracts.</p></div>', '<div><div className="eyebrow">YOUR RECORDS</div><h1>Your records</h1><p>Keep your scans, pathology, treatment records, labs, and genomic reports together in one place.</p></div>', "records header")
s = repl(s, 'New scan reports are best added through the guided New Scan flow.', 'New scan reports are best added from the Trajectory page.', "records new-scan guidance")
s = repl(s, 'Upload the first PDF to begin building a source-grounded cancer history.', 'Upload your first PDF to start building your cancer history.', "records empty copy")
save(p, old, s)

p, s = load("apps/oncotwin-web/app/app/verify/page.tsx")
old = s
s = repl(
    s,
    '<header className="page-head"><div><div className="eyebrow">REVIEW IMPORTANT FACTS</div><h1>Confirm only what needs a second look</h1><p>High-impact, uncertain, or conflicting extracted facts pause here before they enter the committed patient state or affect a research forecast.</p></div>{facts ? <span className="counter">{facts.length} to review</span> : null}</header>',
    '<header className="page-head"><div><div className="eyebrow">REVIEW</div><h1>Check anything that needs a second look</h1><p>Some important details need your confirmation before OncoTwin adds them to your cancer history.</p></div>{facts ? <span className="counter">{facts.length} to review</span> : null}</header>',
    "review header",
)
s = repl(s, 'Important facts with sufficient source support can enter the patient state automatically. Ambiguous or conflicting items will appear here.', 'If OncoTwin finds something uncertain or conflicting, it will appear here for you to check.', "review empty copy")
save(p, old, s)

# -----------------------------------------------------------------------------
# 6) Support component and chart labels: patient-friendly defaults.
# -----------------------------------------------------------------------------
p, s = load("apps/oncotwin-web/components/ModelSupport.tsx")
old = s
s = repl(
    s,
    "{status === 'STANDARD' ? 'Standard support' : status === 'LIMITED' ? 'Limited support' : status === 'UNAVAILABLE' ? 'Forecast unavailable' : status}",
    "{status === 'STANDARD' ? 'Enough information' : status === 'LIMITED' ? 'Limited information' : status === 'UNAVAILABLE' ? 'Not enough information' : status}",
    "model support labels",
)
s = repl(
    s,
    'No additional rule-based limitation was triggered. This still remains a research-model estimate awaiting independent external confirmation.',
    'OncoTwin has enough information to show this trajectory.',
    "model support default copy",
)
save(p, old, s)

p, s = load("apps/oncotwin-web/components/ForecastChart.tsx")
old = s
s = repl(
    s,
    'aria-label="Research-model progression-free survival estimate over 24 months, before and after adding the newest scan evidence"',
    'aria-label="Progression-free outlook over 24 months, before and after the newest scan"',
    "forecast chart aria label",
)
save(p, old, s)

print("PATCHED_FILES=")
for path in changed:
    print(f"  {path}")
PY

  echo
  echo "=== PATIENT-COPY CONTRACT CHECK ==="
  python - <<'PY'
from pathlib import Path

root = Path("apps/oncotwin-web")
files = list(root.rglob("*.tsx"))
text = "\n".join(p.read_text() for p in files)

required = [
    "Research release",
    "Your cancer right now",
    "How your outlook is changing",
    "Your cancer timeline",
    "Add your newest scan",
    "Back to trajectory",
]

for phrase in required:
    assert phrase in text, f"Missing required patient-facing phrase: {phrase}"

forbidden = [
    "Independent external confirmation is pending.",
    "Use synthetic or de-identified records in this hosted build.",
    "Missing ≠ negative",
    "What OncoTwin knows right now",
    "Pathology, treatment, scans, disease-site evidence, and genomics appear as dated events rather than isolated chat messages.",
    "Once your record includes a dated metastatic-disease anchor and a verified scan, OncoTwin can run the frozen research model.",
    "This is a research-model estimate, not a treatment recommendation. Independent external confirmation is still pending.",
    "PRE is the model state before the newest scan.",
    "History-only model state",
    "Frozen selected update",
    "The locked update strength is α",
]

for phrase in forbidden:
    assert phrase not in text, f"Old/jargony patient copy still present: {phrase}"

shell = Path("apps/oncotwin-web/components/AppShell.tsx").read_text()
assert "'/app/forecast'" in shell and "Trajectory" in shell
assert "'/app/update'" not in shell, "Add Newest Scan is still a top-level nav item"

# State hashes remain legitimate internal API/model data, but must not be rendered
# as the old patient-facing STATE XXXXXXXX badge or scan-flow identifier.
overview = Path("apps/oncotwin-web/app/app/page.tsx").read_text()
update = Path("apps/oncotwin-web/app/app/update/page.tsx").read_text()
assert 'className="state-hash"' not in overview
assert "<span>Current state</span>" not in update
assert "<p>State {" not in update

print("PATIENT_COPY_CONTRACT_PASS")
PY

  echo
  echo "=== QUICK COPY SEARCH ==="
  grep -RniE \
    'Research release|Your cancer right now|How your outlook is changing|Add your newest scan|Your cancer timeline|Independent external confirmation|Missing ≠ negative|History-only model state|Frozen selected update|STATE \{' \
    apps/oncotwin-web/app apps/oncotwin-web/components \
    || true

  echo
  echo "=== FRONTEND PRODUCTION BUILD ==="
  (
    cd apps/oncotwin-web
    npm run build
  )

  echo
  echo "PATIENT_COPY_POLISH_PASS"
  echo "ARTIFACT_DIR=$ART"
)
RC=$?

echo
echo "POLISH_CHILD_EXIT_CODE=$RC"
if [ "$RC" -eq 0 ]; then
  echo "COMMAND_STATUS=PASS"
else
  echo "COMMAND_STATUS=FAIL"
fi
echo "Interactive Slurm shell preserved."
true
