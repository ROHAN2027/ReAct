/* ==========================================================================
   ReAct Assistant — Client Application
   Handles chat interactions, SSE streaming, and DOM rendering.
   ========================================================================== */

(() => {
    'use strict';

    // -----------------------------------------------------------------------
    // DOM References
    // -----------------------------------------------------------------------

    const messagesEl    = document.getElementById('messages');
    const chatContainer = document.getElementById('chat-container');
    const userInput     = document.getElementById('user-input');
    const sendBtn       = document.getElementById('send-btn');
    const welcomeScreen = document.getElementById('welcome-screen');
    const memoryCount   = document.getElementById('memory-count');
    const memoryBadge   = document.getElementById('memory-indicator');
    const sessionIdText = document.getElementById('session-id-text');

    let isProcessing = false;

    // -----------------------------------------------------------------------
    // Init
    // -----------------------------------------------------------------------

    function init() {
        sendBtn.addEventListener('click', handleSend);
        userInput.addEventListener('input', handleInputChange);
        userInput.addEventListener('keydown', handleKeydown);

        // Capability chips
        document.querySelectorAll('.chip').forEach(chip => {
            chip.addEventListener('click', () => {
                const query = chip.getAttribute('data-query');
                if (query) {
                    userInput.value = query;
                    handleInputChange();
                    handleSend();
                }
            });
        });

        fetchMemoryStats();
        autoResizeTextarea();
    }

    // -----------------------------------------------------------------------
    // Fetch initial stats
    // -----------------------------------------------------------------------

    async function fetchMemoryStats() {
        try {
            const res = await fetch('/api/memory/stats');
            const data = await res.json();
            memoryCount.textContent = data.total_memories || 0;
            if (data.session_id) {
                sessionIdText.textContent = data.session_id.substring(0, 8);
            }
        } catch (e) {
            console.warn('Could not fetch memory stats:', e);
        }
    }

    // -----------------------------------------------------------------------
    // Input Handling
    // -----------------------------------------------------------------------

    function handleInputChange() {
        const hasText = userInput.value.trim().length > 0;
        sendBtn.disabled = !hasText || isProcessing;
        autoResizeTextarea();
    }

    function handleKeydown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!sendBtn.disabled) handleSend();
        }
    }

    function autoResizeTextarea() {
        userInput.style.height = 'auto';
        userInput.style.height = Math.min(userInput.scrollHeight, 150) + 'px';
    }

    // -----------------------------------------------------------------------
    // Send Message
    // -----------------------------------------------------------------------

    async function handleSend() {
        const text = userInput.value.trim();
        if (!text || isProcessing) return;

        isProcessing = true;
        sendBtn.disabled = true;

        // Hide welcome screen
        if (welcomeScreen) {
            welcomeScreen.style.display = 'none';
        }

        // Render user message
        renderUserMessage(text);
        userInput.value = '';
        userInput.style.height = 'auto';
        scrollToBottom();

        // Show typing indicator
        const typingEl = showTypingIndicator();

        // Current agent message container (built incrementally)
        let agentMessageEl = null;
        let toolContainer  = null;
        let currentToolCard = null;

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text }),
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop(); // keep incomplete line in buffer

                for (const line of lines) {
                    if (line.startsWith('event:')) {
                        var eventType = line.substring(6).trim();
                    } else if (line.startsWith('data:') && eventType) {
                        const dataStr = line.substring(5).trim();
                        if (!dataStr) continue;

                        try {
                            const data = JSON.parse(dataStr);
                            handleSSEEvent(eventType, data);
                        } catch (parseErr) {
                            console.warn('Failed to parse SSE data:', dataStr);
                        }
                        eventType = null;
                    }
                }
            }

        } catch (err) {
            console.error('Chat error:', err);
            removeTypingIndicator(typingEl);
            renderAgentMessage('⚠️ An error occurred while contacting the server. Please try again.');
        }

        removeTypingIndicator(typingEl);
        isProcessing = false;
        handleInputChange();
        scrollToBottom();

        // -- SSE Event Handler (closure over state) --
        function handleSSEEvent(type, data) {
            switch (type) {
                case 'thinking':
                    // Already showing typing indicator
                    break;

                case 'tool_call':
                    removeTypingIndicator(typingEl);
                    if (!agentMessageEl) {
                        agentMessageEl = createAgentMessageShell();
                        toolContainer = agentMessageEl.querySelector('.tool-calls-container');
                    }
                    currentToolCard = renderToolCall(toolContainer, data.name, data.args);
                    scrollToBottom();

                    // Trigger memory badge glow if memory tool
                    if (data.name === 'search_long_term_memory') {
                        triggerMemoryGlow();
                    }
                    break;

                case 'tool_result':
                    if (currentToolCard) {
                        updateToolResult(currentToolCard, data.name, data.result);
                    }
                    scrollToBottom();
                    break;

                case 'answer':
                    removeTypingIndicator(typingEl);
                    if (agentMessageEl) {
                        // Append answer content to existing agent message
                        const contentDiv = agentMessageEl.querySelector('.message-content');
                        contentDiv.innerHTML = renderMarkdown(data.content);
                        contentDiv.style.display = 'block';
                    } else {
                        renderAgentMessage(data.content);
                    }
                    scrollToBottom();
                    break;

                case 'memory_saved':
                    renderStatusMessage('Interaction saved to memory', 'memory');
                    fetchMemoryStats();
                    scrollToBottom();
                    break;

                case 'error':
                    removeTypingIndicator(typingEl);
                    renderAgentMessage('⚠️ ' + (data.detail || 'Unknown error'));
                    break;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Render: User Message
    // -----------------------------------------------------------------------

    function renderUserMessage(text) {
        const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const el = createElement('div', 'message user', `
            <div class="message-avatar">U</div>
            <div class="message-body">
                <div class="message-content">${escapeHtml(text)}</div>
                <div class="message-timestamp">${time}</div>
            </div>
        `);
        messagesEl.appendChild(el);
    }

    // -----------------------------------------------------------------------
    // Render: Agent Message
    // -----------------------------------------------------------------------

    function renderAgentMessage(content) {
        const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const el = createElement('div', 'message agent', `
            <div class="message-avatar">R</div>
            <div class="message-body">
                <div class="tool-calls-container"></div>
                <div class="message-content">${renderMarkdown(content)}</div>
                <div class="message-timestamp">${time}</div>
            </div>
        `);
        messagesEl.appendChild(el);
        return el;
    }

    function createAgentMessageShell() {
        const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const el = createElement('div', 'message agent', `
            <div class="message-avatar">R</div>
            <div class="message-body">
                <div class="tool-calls-container"></div>
                <div class="message-content" style="display:none;"></div>
                <div class="message-timestamp">${time}</div>
            </div>
        `);
        messagesEl.appendChild(el);
        return el;
    }

    // -----------------------------------------------------------------------
    // Render: Tool Call Cards
    // -----------------------------------------------------------------------

    const TOOL_EMOJIS = {
        'arxiv': '📄',
        'wikipedia': '📚',
        'tavily_search_results_json': '🌐',
        'news_search': '📰',
        'python_repl': '🐍',
        'search_long_term_memory': '🧠',
        'get_current_datetime': '🕐',
        'read_file': '📖',
        'write_file': '✏️',
        'list_directory': '📂',
    };

    function renderToolCall(container, name, args) {
        const emoji = TOOL_EMOJIS[name] || '🔧';
        const isMemory = name === 'search_long_term_memory';
        const argsStr = typeof args === 'object' ? JSON.stringify(args, null, 2) : String(args);

        const card = createElement('div', `tool-card${isMemory ? ' memory-tool' : ''}`, `
            <div class="tool-card-header">
                <div class="tool-icon calling">${emoji}</div>
                <span class="tool-name">${escapeHtml(name)}</span>
                <span class="tool-status-text" style="font-size:0.7rem;color:var(--warning);">calling…</span>
                <span class="tool-chevron">▶</span>
            </div>
            <div class="tool-card-body">
                <div class="tool-section">
                    <div class="tool-section-label">Input</div>
                    <div class="tool-section-content">${escapeHtml(argsStr)}</div>
                </div>
                <div class="tool-result-section"></div>
            </div>
        `);

        // Toggle expand/collapse
        card.querySelector('.tool-card-header').addEventListener('click', () => {
            card.classList.toggle('expanded');
        });

        container.appendChild(card);
        return card;
    }

    function updateToolResult(card, name, result) {
        const icon = card.querySelector('.tool-icon');
        icon.classList.remove('calling');
        icon.classList.add('done');

        const statusText = card.querySelector('.tool-status-text');
        statusText.textContent = 'done';
        statusText.style.color = 'var(--success)';

        const resultSection = card.querySelector('.tool-result-section');
        resultSection.innerHTML = `
            <div class="tool-section">
                <div class="tool-section-label">Output</div>
                <div class="tool-section-content">${escapeHtml(result)}</div>
            </div>
        `;
    }

    // -----------------------------------------------------------------------
    // Render: Typing Indicator
    // -----------------------------------------------------------------------

    function showTypingIndicator() {
        const el = createElement('div', 'typing-indicator', `
            <div class="message-avatar" style="background:var(--bg-surface);border:1px solid var(--glass-border);color:var(--text-accent);">R</div>
            <div>
                <div class="typing-dots">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        `);
        el.id = 'typing-indicator';
        messagesEl.appendChild(el);
        scrollToBottom();
        return el;
    }

    function removeTypingIndicator(el) {
        if (el && el.parentNode) {
            el.parentNode.removeChild(el);
        }
    }

    // -----------------------------------------------------------------------
    // Render: Status Messages
    // -----------------------------------------------------------------------

    function renderStatusMessage(text, type) {
        const el = createElement('div', `status-message ${type || ''}`, `
            <span class="status-dot"></span>
            <span>${escapeHtml(text)}</span>
        `);
        messagesEl.appendChild(el);
    }

    // -----------------------------------------------------------------------
    // Memory Badge Glow
    // -----------------------------------------------------------------------

    function triggerMemoryGlow() {
        memoryBadge.classList.remove('active');
        void memoryBadge.offsetWidth; // trigger reflow
        memoryBadge.classList.add('active');
    }

    // -----------------------------------------------------------------------
    // Markdown Renderer (lightweight inline)
    // -----------------------------------------------------------------------

    function renderMarkdown(text) {
        if (!text) return '';

        let html = escapeHtml(text);

        // Code blocks (``` ... ```)
        html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
            return `<pre><code>${code.trim()}</code></pre>`;
        });

        // Inline code
        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

        // Bold
        html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

        // Italic
        html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

        // Headers
        html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
        html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
        html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

        // Blockquotes
        html = html.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');

        // Unordered lists
        html = html.replace(/^[\-\*] (.+)$/gm, '<li>$1</li>');
        html = html.replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>');

        // Links
        html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

        // Paragraphs (double newline)
        html = html.replace(/\n\n/g, '</p><p>');
        html = '<p>' + html + '</p>';

        // Single newlines → <br>
        html = html.replace(/\n/g, '<br>');

        // Clean up empty paragraphs
        html = html.replace(/<p>\s*<\/p>/g, '');

        return html;
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    function createElement(tag, className, innerHTML) {
        const el = document.createElement(tag);
        if (className) el.className = className;
        if (innerHTML) el.innerHTML = innerHTML;
        return el;
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    function scrollToBottom() {
        requestAnimationFrame(() => {
            chatContainer.scrollTo({
                top: chatContainer.scrollHeight,
                behavior: 'smooth'
            });
        });
    }

    // -----------------------------------------------------------------------
    // Boot
    // -----------------------------------------------------------------------

    document.addEventListener('DOMContentLoaded', init);

})();
