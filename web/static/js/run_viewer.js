/**
 * StreamSociety - Run Viewer JavaScript
 * Handles timeline scrubber, chat replay, and metrics update
 */

'use strict';

let runData = null;

// Fetch run data from API
async function fetchRunData() {
    try {
        const response = await fetch(`/runs/${RUN_ID}/data`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        runData = await response.json();
        initViewer();
    } catch (err) {
        console.error('Failed to fetch run data:', err);
        showToast('Failed to load run data', 'error');
    }
}

// Initialize viewer after data is loaded
function initViewer() {
    if (!runData || !runData.turns) return;

    const turns = runData.turns;
    const scrubber = document.getElementById('timeline-scrubber');
    const totalTurnsEl = document.getElementById('total-turns');

    if (totalTurnsEl) {
        totalTurnsEl.textContent = turns.length;
    }

    if (scrubber) {
        scrubber.max = Math.max(0, turns.length - 1);
        scrubber.addEventListener('input', function() {
            const turnIdx = parseInt(this.value);
            document.getElementById('scrubber-display').textContent = turnIdx;
            renderTurn(turnIdx);
        });
    }

    // Build timeline marks
    buildTimelineMarks(turns.length);

    // Render first turn
    renderTurn(0);
}

// Build tick marks on the timeline
function buildTimelineMarks(numTurns) {
    const marksEl = document.getElementById('timeline-marks');
    if (!marksEl || numTurns === 0) return;

    const markCount = Math.min(numTurns, 20);
    const step = Math.floor(numTurns / markCount);
    let html = '<div style="display:flex; justify-content:space-between; margin-top:4px;">';
    for (let i = 0; i <= numTurns; i += step) {
        html += `<span style="font-size:0.65rem; color:var(--text-muted);">${i}</span>`;
    }
    html += '</div>';
    marksEl.innerHTML = html;
}

// Render a specific turn
function renderTurn(turnIdx) {
    if (!runData || !runData.turns) return;
    const turns = runData.turns;
    if (turnIdx < 0 || turnIdx >= turns.length) return;

    const turn = turns[turnIdx];

    // Update turn counter
    const turnDisplay = document.getElementById('turn-display');
    if (turnDisplay) turnDisplay.textContent = turnIdx;

    // Update current topic
    const topicEl = document.getElementById('current-topic');
    if (topicEl && turn.streamer_response) {
        topicEl.textContent = turn.streamer_response.metadata?.topic || '-';
    } else if (topicEl && turn.selected_comment) {
        topicEl.textContent = turn.selected_comment.topic || '-';
    }

    // Update streamer response
    const responseTextEl = document.getElementById('response-text');
    if (responseTextEl) {
        if (turn.streamer_response) {
            responseTextEl.textContent = turn.streamer_response.response_text;
            responseTextEl.style.color = 'var(--text-primary)';
        } else {
            responseTextEl.textContent = '-- このターンは応答なし --';
            responseTextEl.style.color = 'var(--text-muted)';
        }
    }

    // Update metrics
    updateMetrics(turn);

    // Update chat
    renderChat(turns, turnIdx);
}

// Update metrics bar
function updateMetrics(turn) {
    const metrics = turn.metrics || {};

    const engEl = document.getElementById('m-engagement');
    if (engEl) {
        const candidates = metrics.num_candidates || 0;
        const viewers = runData.summary?.num_viewers || 1;
        engEl.textContent = formatFloat(candidates / viewers, 2);
    }

    const sentEl = document.getElementById('m-sentiment');
    if (sentEl) {
        const val = metrics.avg_sentiment;
        sentEl.textContent = val !== undefined ? formatFloat(val, 3) : '-';
        sentEl.style.color = val > 0.3 ? 'var(--accent-green)' :
                             val < -0.3 ? 'var(--accent-red)' :
                             'var(--accent-blue)';
    }

    const candEl = document.getElementById('m-candidates');
    if (candEl) {
        candEl.textContent = metrics.num_candidates || 0;
    }
}

// Render chat panel for current turn
function renderChat(turns, currentTurnIdx) {
    const chatEl = document.getElementById('chat-messages');
    if (!chatEl) return;

    // Show all comments up to current turn (last 30 max)
    let allComments = [];
    const startTurn = Math.max(0, currentTurnIdx - 29);
    for (let i = startTurn; i <= currentTurnIdx; i++) {
        const turn = turns[i];
        if (!turn.comment_candidates) continue;
        for (const comment of turn.comment_candidates) {
            allComments.push({
                comment,
                isSelected: turn.selected_comment?.comment_id === comment.comment_id,
                turn: i
            });
        }
    }

    if (allComments.length === 0) {
        chatEl.innerHTML = '<div class="chat-placeholder">このターンにコメントはありません</div>';
        return;
    }

    const html = allComments.map(({ comment, isSelected, turn: t }) => {
        const sentIcon = sentimentIcon(comment.sentiment);
        const questionMark = comment.question_flag ? ' ❓' : '';
        const selectedBadge = isSelected ? '<span class="comment-selected-badge">&#10003; Selected</span>' : '';
        const turnLabel = t < currentTurnIdx ? `<span style="font-size:0.65rem;color:var(--text-muted);">T${t}</span>` : '';

        return `
            <div class="comment-bubble ${isSelected ? 'selected' : ''}">
                <div class="comment-header">
                    ${turnLabel}
                    <span class="comment-name">${escapeHtml(comment.viewer_id)}</span>
                    <span class="comment-group">${escapeHtml(comment.persona_group)}</span>
                    <span class="comment-sentiment">${sentIcon}${questionMark}</span>
                </div>
                <div class="comment-text">${escapeHtml(comment.text)}</div>
                ${selectedBadge}
            </div>
        `;
    }).join('');

    chatEl.innerHTML = html;
    scrollChatToBottom(chatEl);
}

// Escape HTML to prevent XSS
function escapeHtml(str) {
    if (!str) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    if (typeof RUN_ID !== 'undefined') {
        fetchRunData();
    }
});
