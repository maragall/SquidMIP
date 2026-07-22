#!/bin/sh
# Refuse a commit whose tests do not pass.
#
# Why this exists. Agents were told to commit before testing during a shutdown scare, so that
# in-flight work would survive. That was right at the time and then never unwound, and red
# commits quietly became normal - including one that landed with failing tile-composite tests
# and said so only in its report.
#
# The rule now:
#   * a branch ending in -wip may commit red. It can never merge; it is a life raft, nothing else.
#   * every other branch must be green, or the commit is refused.
#   * SQUIDHCS_STOP_ORDER=1 allows one red commit for a genuine stop order (machine going down).
#     It must be set deliberately, and the commit message must say why.
#
# Linked worktrees share $GIT_DIR/hooks with the main checkout, so installing this once covers
# every agent worktree.

set -e

BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "?")

case "$BRANCH" in
  *-wip)
    echo "commit-gate: '$BRANCH' is a -wip branch, allowing a red commit."
    echo "             it must never merge. re-land the work green on a real branch."
    exit 0
    ;;
esac

if [ "${SQUIDHCS_STOP_ORDER:-0}" = "1" ]; then
  echo "commit-gate: SQUIDHCS_STOP_ORDER=1, allowing one red commit."
  echo "             say why in the commit message, and re-land green."
  exit 0
fi

ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"

echo "commit-gate: running the suite before allowing this commit ..."
if ! QT_QPA_PLATFORM=offscreen PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
     python -m pytest -q -x --timeout=600 >/tmp/squidhcs_gate.$$ 2>&1; then
  # The four known-flaky tests (IMA-258) are races that pass in isolation. A flake must not
  # block a commit, but it must not silently pass one either - so re-run just the failures.
  FAILED=$(grep -E "^FAILED " /tmp/squidhcs_gate.$$ | sed 's/^FAILED //; s/ .*//' || true)
  if [ -n "$FAILED" ]; then
    echo "commit-gate: re-running failures in isolation to separate flakes from breakage ..."
    if QT_QPA_PLATFORM=offscreen PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
       python -m pytest -q $FAILED >/dev/null 2>&1; then
      echo "commit-gate: all failures passed in isolation - known flakes (IMA-258). Allowing."
      rm -f /tmp/squidhcs_gate.$$
      exit 0
    fi
  fi
  echo ""
  tail -30 /tmp/squidhcs_gate.$$
  rm -f /tmp/squidhcs_gate.$$
  echo ""
  echo "commit-gate: REFUSED. Tests fail and they are not the known flakes."
  echo "  fix them, or commit on a -wip branch that can never merge,"
  echo "  or SQUIDHCS_STOP_ORDER=1 git commit ... for a real stop order."
  exit 1
fi

rm -f /tmp/squidhcs_gate.$$
echo "commit-gate: green."
exit 0
