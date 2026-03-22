#!/bin/bash
# Local data accumulator for HazardPulse tornado predictions.
# Runs the scoring pipeline every 2 hours during convective hours (12Z-03Z).
# Builds up the prediction ledger for verification before going public.
#
# Usage: nohup bash scripts/accumulate.sh &
#
# To stop: kill $(cat /tmp/hazardpulse_accumulate.pid)

echo $$ > /tmp/hazardpulse_accumulate.pid
echo "HazardPulse accumulator started (PID $$)"

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

export HAZARDPULSE_HRRR_CACHE="C:/Users/Josh/Projects/Coherence_Energy_Labs_Website/data/evidence/labs/hazard_prediction_lab/cache/hrrr"
export HAZARDPULSE_PROBSEVERE_CACHE="C:/Users/Josh/Projects/Coherence_Energy_Labs_Website/data/evidence/labs/hazard_prediction_lab/cache/probsevere"

while true; do
    HOUR=$(date -u +%H)

    # Only run during convective hours (12Z-03Z = 12,13,...,23,0,1,2,3)
    if [ "$HOUR" -ge 12 ] || [ "$HOUR" -le 3 ]; then
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Running scoring pipeline..."
        python -u scripts/fetch_and_score_tornado.py 2>&1 | tail -5

        # Auto-commit if ledger changed
        if git diff --quiet dist/data/tornado-ledger.jsonl 2>/dev/null; then
            echo "  No changes."
        else
            git add dist/data/live-tornadoes.json dist/data/live-pulse.json dist/data/tornado-ledger.jsonl
            git commit -m "data: tornado scoring $(date -u +%Y-%m-%dT%H:%MZ)" --quiet
            git push origin main --quiet 2>/dev/null && echo "  Pushed." || echo "  Push failed (will retry)."
        fi
    else
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Outside convective hours, sleeping."
    fi

    # Sleep 2 hours
    sleep 7200
done
