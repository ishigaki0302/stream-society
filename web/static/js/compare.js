/**
 * StreamSociety - Compare Page JavaScript
 * Handles multi-run comparison and ASCII chart rendering
 */

'use strict';

// Load comparison data from selected runs
async function loadComparison() {
    const checkboxes = document.querySelectorAll('.run-checkbox:checked');
    if (checkboxes.length === 0) {
        showToast('Please select at least one run', 'error');
        return;
    }

    const runIds = Array.from(checkboxes).map(cb => cb.value);
    const params = new URLSearchParams({ run_ids: runIds.join(',') });

    try {
        const response = await fetch(`/compare/data?${params}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();

        renderComparisonTable(data.runs);
        renderMetricsChart(data.runs);
    } catch (err) {
        console.error('Failed to load comparison:', err);
        showToast('Failed to load comparison data', 'error');
    }
}

// Render comparison table
function renderComparisonTable(runs) {
    const resultsEl = document.getElementById('compare-results');
    if (!resultsEl) return;

    if (!runs || runs.length === 0) {
        resultsEl.innerHTML = '<div class="empty-state"><p>No data available</p></div>';
        return;
    }

    const metrics = [
        { key: 'policy', label: 'Policy' },
        { key: 'num_turns', label: 'Turns' },
        { key: 'num_viewers', label: 'Viewers' },
        { key: 'total_comments', label: 'Total Comments' },
        { key: 'engagement_proxy', label: 'Engagement', fmt: v => formatFloat(v, 4) },
        { key: 'unique_participant_rate', label: 'Unique Rate', fmt: v => formatFloat(v, 4) },
        { key: 'topic_diversity', label: 'Topic Diversity', fmt: v => formatFloat(v, 4) },
        { key: 'safety_rate', label: 'Safety Rate', fmt: v => formatFloat(v, 4) },
        { key: 'sentiment_shift', label: 'Sentiment Shift', fmt: v => (v >= 0 ? '+' : '') + formatFloat(v, 4) },
    ];

    let html = `<table class="compare-table">
        <thead>
            <tr>
                <th>Metric</th>
                ${runs.map(r => `<th>${escapeHtml(r.run_id?.slice(0, 16) + '...' || '-')}</th>`).join('')}
            </tr>
        </thead>
        <tbody>`;

    metrics.forEach(m => {
        const bestVal = findBest(runs, m.key);
        html += `<tr>
            <td style="font-weight:600;color:var(--text-secondary)">${m.label}</td>
            ${runs.map(r => {
                const val = r[m.key];
                const formatted = m.fmt ? m.fmt(val) : (val !== undefined ? String(val) : '-');
                const isBest = val === bestVal && typeof val === 'number';
                return `<td style="${isBest ? 'color:var(--accent-green);font-weight:700' : ''}">${formatted}</td>`;
            }).join('')}
        </tr>`;
    });

    html += '</tbody></table>';
    resultsEl.innerHTML = html;
}

// Find the best (highest) value for numeric metrics
function findBest(runs, key) {
    const vals = runs.map(r => r[key]).filter(v => typeof v === 'number');
    if (vals.length === 0) return null;
    // For sentiment_shift and safety, higher is better; for toxicity lower is better
    return Math.max(...vals);
}

// Render ASCII-style metrics chart
function renderMetricsChart(runs) {
    const chartContainer = document.getElementById('chart-container');
    const chartEl = document.getElementById('metrics-chart');
    if (!chartEl || !chartContainer) return;

    if (!runs || runs.length === 0) {
        chartContainer.style.display = 'none';
        return;
    }

    chartContainer.style.display = 'block';

    const metricsToChart = [
        { key: 'engagement_proxy', label: 'Engagement' },
        { key: 'safety_rate', label: 'Safety Rate' },
        { key: 'unique_participant_rate', label: 'Unique Rate' },
        { key: 'topic_diversity', label: 'Topic Diversity' },
    ];

    const barWidth = 20;
    let chart = '';

    metricsToChart.forEach(m => {
        const vals = runs.map(r => parseFloat(r[m.key]) || 0);
        const maxVal = Math.max(...vals, 0.001);

        chart += `\n${m.label}\n`;
        runs.forEach((r, i) => {
            const policy = (r.policy || 'unknown').padEnd(18);
            const normalized = vals[i] / maxVal;
            const barLen = Math.round(normalized * barWidth);
            const bar = '█'.repeat(barLen) + '░'.repeat(barWidth - barLen);
            const valStr = formatFloat(vals[i], 4).padStart(7);
            chart += `  ${policy} |${bar}| ${valStr}\n`;
        });
    });

    chartEl.textContent = chart;
}

// Escape HTML
function escapeHtml(str) {
    if (!str) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}
