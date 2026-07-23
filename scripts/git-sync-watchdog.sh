#!/usr/bin/env bash
#
# git-sync-watchdog.sh — auto fast-forward pull + conflict notifier for a
# two-developer shared-branch workflow (Julio + Spencer push to the same
# branches of the same repo).
#
# Behavior, per loop iteration (default every 60s, override with WATCH_INTERVAL):
#   1. git fetch (guarded; a network failure is logged and the loop continues).
#   2. Compare local HEAD to its tracked upstream:
#        - up to date            -> do nothing.
#        - upstream is AHEAD only -> fast-forward pull (unless working tree is
#                                    dirty, in which case notify + do nothing).
#        - branches have DIVERGED -> DO NOT merge/rebase. Notify (macOS) + log
#                                    both SHAs so a human resolves it.
#
# It NEVER auto-resolves a conflict. It only ever fast-forwards. Anything that
# would require a merge/rebase is surfaced to the user, not performed.
#
# Environment:
#   WATCH_INTERVAL   seconds between iterations (default 60)
#   WATCH_ONCE=1     run a single iteration and exit (for testing / cron)
#   REPO_DIR         repo to watch (default: the script's own repo root)
#
# Usage:
#   bash scripts/git-sync-watchdog.sh          # loop forever
#   WATCH_ONCE=1 bash scripts/git-sync-watchdog.sh   # one iteration, then exit
#
# Safe to Ctrl-C at any time.

set -u

# --- configuration ----------------------------------------------------------

WATCH_INTERVAL="${WATCH_INTERVAL:-60}"
WATCH_ONCE="${WATCH_ONCE:-0}"

# Resolve the repo dir: explicit REPO_DIR wins, else the dir containing this
# script's parent (scripts/ lives at repo root).
if [ -n "${REPO_DIR:-}" ]; then
    REPO="$REPO_DIR"
else
    _self="${BASH_SOURCE[0]:-$0}"
    _selfdir="$(cd "$(dirname "$_self")" >/dev/null 2>&1 && pwd)"
    REPO="$(cd "$_selfdir/.." >/dev/null 2>&1 && pwd)"
fi

# --- helpers ----------------------------------------------------------------

log() {
    # timestamped stderr line
    printf '%s [git-sync] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

notify() {
    # macOS notification; silently no-op on non-macOS or if osascript missing.
    local title="$1" msg="$2"
    if command -v osascript >/dev/null 2>&1; then
        # Escape double quotes for AppleScript string literals.
        local esc_msg esc_title
        esc_msg=${msg//\"/\\\"}
        esc_title=${title//\"/\\\"}
        osascript -e "display notification \"$esc_msg\" with title \"$esc_title\"" >/dev/null 2>&1 || true
    fi
}

# Run git in the repo. Never lets a failure kill the loop; caller inspects rc.
git_repo() {
    git -C "$REPO" "$@"
}

# --- one iteration ----------------------------------------------------------

run_once() {
    # Validate repo each iteration (cheap; survives the repo going away).
    if ! git_repo rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        log "ERROR: '$REPO' is not a git work tree; skipping."
        return 1
    fi

    local branch upstream
    branch="$(git_repo rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
    upstream="$(git_repo rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || echo '')"
    if [ -z "$upstream" ]; then
        log "branch '$branch' has no tracking upstream; nothing to do."
        return 0
    fi

    # 1. Fetch. Transient failure (network) -> log + continue, never fatal.
    if ! git_repo fetch --quiet 2>/dev/null; then
        log "fetch failed (transient/network?) on '$branch'; will retry next cycle."
        return 0
    fi

    # 2. Compare local vs upstream.
    local local_sha remote_sha base
    local_sha="$(git_repo rev-parse HEAD 2>/dev/null || echo '')"
    remote_sha="$(git_repo rev-parse '@{u}' 2>/dev/null || echo '')"
    base="$(git_repo merge-base HEAD '@{u}' 2>/dev/null || echo '')"

    if [ -z "$local_sha" ] || [ -z "$remote_sha" ] || [ -z "$base" ]; then
        log "could not resolve SHAs on '$branch'; skipping this cycle."
        return 0
    fi

    if [ "$local_sha" = "$remote_sha" ]; then
        log "up to date on '$branch' (${local_sha:0:7})."
        return 0
    fi

    local ahead behind
    # commits local has that upstream doesn't (local ahead)
    ahead="$(git_repo rev-list --count '@{u}..HEAD' 2>/dev/null || echo 0)"
    # commits upstream has that local doesn't (local behind)
    behind="$(git_repo rev-list --count 'HEAD..@{u}' 2>/dev/null || echo 0)"

    # Case A: DIVERGED — both sides have unique commits. Conflict risk.
    if [ "$ahead" -gt 0 ] && [ "$behind" -gt 0 ]; then
        log "DIVERGED on '$branch': local=${local_sha:0:7} (+$ahead) upstream=${upstream} ${remote_sha:0:7} (+$behind). NOT auto-merging — resolve manually."
        notify "Git sync: DIVERGED ($branch)" \
            "Local +$ahead / remote +$behind. Manual merge needed. local ${local_sha:0:7} vs ${remote_sha:0:7}"
        return 0
    fi

    # Case B: local is AHEAD only — nothing to pull (Julio has unpushed work).
    if [ "$ahead" -gt 0 ] && [ "$behind" -eq 0 ]; then
        log "local '$branch' is ahead of $upstream by $ahead commit(s); nothing to pull."
        return 0
    fi

    # Case C: upstream is AHEAD only ($behind > 0, $ahead == 0) -> fast-forwardable.
    # But never touch the working tree if there are uncommitted changes.
    local dirty
    dirty="$(git_repo status --porcelain 2>/dev/null || echo 'ERR')"
    if [ -n "$dirty" ]; then
        log "remote advanced by $behind on '$branch' but working tree is DIRTY — commit/stash then pull. Doing nothing."
        notify "Git sync: pull blocked ($branch)" \
            "Remote advanced +$behind but working tree is dirty. Commit or stash, then pull."
        return 0
    fi

    # Clean tree + fast-forward available -> pull.
    if git_repo pull --ff-only --quiet 2>/dev/null; then
        local new_sha
        new_sha="$(git_repo rev-parse HEAD 2>/dev/null || echo '')"
        log "fast-forwarded '$branch' by $behind commit(s) from $upstream (now ${new_sha:0:7})."
        notify "Git sync: pulled $behind commit(s)" \
            "$branch fast-forwarded from $upstream."
    else
        # ff-only refused (race: someone diverged between our check and pull).
        log "ff-only pull refused on '$branch' (raced divergence?); NOT forcing. Will re-evaluate next cycle."
        notify "Git sync: pull refused ($branch)" \
            "Fast-forward no longer possible; manual check needed."
    fi
    return 0
}

# --- main loop --------------------------------------------------------------

RUNNING=1
on_interrupt() {
    RUNNING=0
    log "interrupt received; exiting cleanly."
    exit 0
}
trap on_interrupt INT TERM

main() {
    log "watchdog starting. repo='$REPO' interval=${WATCH_INTERVAL}s once=${WATCH_ONCE}"

    if [ "$WATCH_ONCE" = "1" ]; then
        run_once
        log "WATCH_ONCE set; single iteration complete, exiting."
        return 0
    fi

    while [ "$RUNNING" = "1" ]; do
        run_once || true
        # Sleep in the foreground but stay Ctrl-C responsive.
        sleep "$WATCH_INTERVAL" &
        wait $! 2>/dev/null || true
    done
}

main
