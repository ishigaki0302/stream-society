/**
 * StreamSociety - Main JavaScript
 * Shared utilities and helpers
 */

'use strict';

// Utility: format a float to N decimal places
function formatFloat(val, decimals = 3) {
    if (val === null || val === undefined) return '-';
    return parseFloat(val).toFixed(decimals);
}

// Utility: sentiment icon based on value
function sentimentIcon(val) {
    if (val === null || val === undefined) return '😐';
    if (val > 0.3) return '😊';
    if (val < -0.3) return '😟';
    return '😐';
}

// Utility: get policy badge HTML
function policyBadge(policy) {
    return `<span class="policy-badge policy-${policy}">${policy}</span>`;
}

// Smooth scroll to bottom of chat
function scrollChatToBottom(el) {
    if (el) {
        el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
    }
}

// Debounce function
function debounce(fn, ms) {
    let timer;
    return function(...args) {
        clearTimeout(timer);
        timer = setTimeout(() => fn.apply(this, args), ms);
    };
}

// Show toast notification
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    toast.style.cssText = `
        position: fixed;
        bottom: 2rem;
        right: 2rem;
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-left: 3px solid var(--accent-${type === 'error' ? 'red' : type === 'success' ? 'green' : 'blue'});
        color: var(--text-primary);
        padding: 0.75rem 1rem;
        border-radius: 8px;
        font-size: 0.875rem;
        z-index: 9999;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        animation: slideIn 0.3s ease;
    `;
    document.body.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transition = 'opacity 0.3s';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// Initialize page-specific scripts based on body data attribute or URL
document.addEventListener('DOMContentLoaded', function() {
    // Highlight active nav link
    const currentPath = window.location.pathname;
    document.querySelectorAll('.nav-link').forEach(link => {
        const href = link.getAttribute('href');
        if (href === currentPath || (href !== '/' && currentPath.startsWith(href))) {
            link.style.color = 'var(--text-primary)';
            link.style.fontWeight = '700';
        }
    });
});
