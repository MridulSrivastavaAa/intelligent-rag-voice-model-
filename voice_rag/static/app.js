/**
 * VoiceRAG Flora • Complete Modern Application Controller
 * High-performance, resilient, and rich interactive logic.
 */

document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const brandHome = document.getElementById('brand-home');
    const systemStatusDot = document.getElementById('system-status-dot');
    const backendInfo = document.getElementById('backend-info');
    const themeToggleBtn = document.getElementById('theme-toggle-btn');
    const themeIcon = document.getElementById('theme-icon');

    // Voice & Input Elements
    const micBtn = document.getElementById('mic-btn');
    const micIcon = document.getElementById('mic-icon');
    const recordingStatus = document.getElementById('recording-status');
    const recordTimer = document.getElementById('record-timer');
    const voiceStatusPill = document.getElementById('voice-status-pill');
    const waveCanvas = document.getElementById('wave-canvas');
    const queryInput = document.getElementById('query-input');
    const clearInputBtn = document.getElementById('clear-input-btn');
    const searchForm = document.getElementById('search-form');
    const sendBtn = document.getElementById('send-btn');
    const uploadAudioBtn = document.getElementById('upload-audio-btn');
    const audioFileInput = document.getElementById('audio-file-input');
    const languageSelect = document.getElementById('language-select');
    const topkSelect = document.getElementById('topk-select');
    const autoSpeakToggle = document.getElementById('auto-speak-toggle');
    const presetsGrid = document.getElementById('presets-grid');

    // Answer & Output Elements
    const answerCard = document.getElementById('answer-card');
    const answerStatusBadge = document.getElementById('answer-status-badge');
    const answerText = document.getElementById('answer-text');
    const citationsList = document.getElementById('citations-list');
    const copyAnswerBtn = document.getElementById('copy-answer-btn');
    const speakBrowserBtn = document.getElementById('speak-browser-btn');

    // Spoken Audio Player Elements
    const audioPlayerBar = document.getElementById('audio-player-bar');
    const playPauseBtn = document.getElementById('play-pause-btn');
    const audioPlayIcon = document.getElementById('audio-play-icon');
    const audioSpeakerLabel = document.getElementById('audio-speaker-label');
    const audioLatencyLabel = document.getElementById('audio-latency-label');
    const audioSeeker = document.getElementById('audio-seeker');
    const audioSpeedBtn = document.getElementById('audio-speed-btn');
    const downloadAudioBtn = document.getElementById('download-audio-btn');
    const audioElement = document.getElementById('audio-element');

    // Guardrails & Timings Elements
    const guardrailsCard = document.getElementById('guardrails-card');
    const guardrailBadgesGrid = document.getElementById('guardrail-badges-grid');
    const stagesCard = document.getElementById('stages-card');
    const stagesWaterfall = document.getElementById('stages-waterfall');
    const totalLatencyBadge = document.getElementById('total-latency-badge');

    // Context Passages Elements
    const passagesContainer = document.getElementById('passages-container');
    const retrievalCountLabel = document.getElementById('retrieval-count-label');

    // Modals & Drawers Elements
    const openDocsBtn = document.getElementById('open-docs-btn');
    const docsModal = document.getElementById('docs-modal');
    const closeDocsModal = document.getElementById('close-docs-modal');
    const docSearchInput = document.getElementById('doc-search-input');
    const docsListContainer = document.getElementById('docs-list-container');

    const openHistoryBtn = document.getElementById('open-history-btn');
    const historyDrawer = document.getElementById('history-drawer');
    const closeHistoryDrawer = document.getElementById('close-history-drawer');
    const clearHistoryBtn = document.getElementById('clear-history-btn');
    const historyListContainer = document.getElementById('history-list-container');

    const toastContainer = document.getElementById('toast-container');

    // --- State Variables ---
    let isRecording = false;
    let mediaRecorder = null;
    let recordedAudioChunks = [];
    let audioStream = null;
    let audioCtx = null;
    let analyser = null;
    let visualizerAnimId = null;
    let recordStartTime = 0;
    let recordTimerInterval = null;
    let speechRecognition = null;
    let playbackSpeeds = [1.0, 1.25, 1.5, 2.0];
    let currentSpeedIndex = 0;
    let allDocuments = [];
    let queryHistory = JSON.parse(localStorage.getItem('voicerag_history') || '[]');

    // --- Initialize Application ---
    initTheme();
    fetchSystemHealth();
    initIdleVisualizer();
    renderHistory();

    // =========================================================================
    // Theme Switcher (Dark / Light)
    // =========================================================================
    function initTheme() {
        const savedTheme = localStorage.getItem('voicerag_theme') || 'dark';
        document.documentElement.setAttribute('data-theme', savedTheme);
        updateThemeIcon(savedTheme);
    }

    themeToggleBtn.addEventListener('click', () => {
        const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('voicerag_theme', newTheme);
        updateThemeIcon(newTheme);
        showToast(`Switched to ${newTheme} mode`, 'info');
    });

    function updateThemeIcon(theme) {
        if (theme === 'dark') {
            themeIcon.className = 'fa-solid fa-sun';
            themeToggleBtn.title = 'Switch to Light Mode';
        } else {
            themeIcon.className = 'fa-solid fa-moon';
            themeToggleBtn.title = 'Switch to Dark Mode';
        }
    }

    // =========================================================================
    // Backend Health & Stats Polling
    // =========================================================================
    async function fetchSystemHealth() {
        try {
            const res = await fetch('/api/health');
            if (res.ok) {
                const data = await res.json();
                systemStatusDot.classList.remove('offline');
                backendInfo.innerHTML = `Online • <strong>${data.generator_backend || 'Ready'}</strong> (${data.docs_count || 0} docs, ${data.chunks_count || 0} chunks)`;
            } else {
                setBackendOffline();
            }
        } catch (e) {
            setBackendOffline();
        }
    }

    function setBackendOffline() {
        systemStatusDot.classList.add('offline');
        backendInfo.textContent = 'Backend Offline / Reconnecting...';
    }

    // Periodic health check every 15 seconds
    setInterval(fetchSystemHealth, 15000);

    // =========================================================================
    // Toast Notification System
    // =========================================================================
    function showToast(message, type = 'info', duration = 3500) {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        let icon = 'fa-circle-info';
        if (type === 'success') icon = 'fa-circle-check';
        if (type === 'error') icon = 'fa-circle-exclamation';

        toast.innerHTML = `
            <i class="fa-solid ${icon}"></i>
            <span>${escapeHtml(message)}</span>
        `;
        toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(10px)';
            toast.style.transition = 'all 0.3s ease';
            setTimeout(() => toast.remove(), 300);
        }, duration);
    }

    // =========================================================================
    // Input Handling & Presets
    // =========================================================================
    queryInput.addEventListener('input', () => {
        clearInputBtn.style.display = queryInput.value.trim() ? 'flex' : 'none';
    });

    clearInputBtn.addEventListener('click', () => {
        queryInput.value = '';
        clearInputBtn.style.display = 'none';
        queryInput.focus();
    });

    presetsGrid.addEventListener('click', (e) => {
        const chip = e.target.closest('.preset-chip');
        if (chip) {
            const query = chip.getAttribute('data-query');
            if (query) {
                queryInput.value = query;
                clearInputBtn.style.display = 'flex';
                handleTextQuerySubmit(query);
            }
        }
    });

    searchForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const text = queryInput.value.trim();
        if (text) {
            handleTextQuerySubmit(text);
        }
    });

    brandHome.addEventListener('click', () => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });

    // =========================================================================
    // Audio File Upload Handling
    // =========================================================================
    uploadAudioBtn.addEventListener('click', () => {
        audioFileInput.click();
    });

    audioFileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;

        showToast(`Selected audio file: ${file.name}`, 'info');
        handleAudioFileUpload(file);
        audioFileInput.value = '';
    });

    async function handleAudioFileUpload(file) {
        setLoading(true, 'Uploading and transcribing audio with Sarvam AI...');
        const formData = new FormData();
        formData.append('audio', file, file.name);
        formData.append('top_k', topkSelect.value);

        await executeQueryRequest(formData);
    }

    // =========================================================================
    // Microphone Recording & Universal Web Audio
    // =========================================================================
    micBtn.addEventListener('click', toggleRecording);

    async function toggleRecording() {
        if (!isRecording) {
            await startRecording();
        } else {
            stopRecording();
        }
    }

    async function startRecording() {
        try {
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error('Microphone recording is not supported in this browser.');
            }

            audioStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });

            // Find best supported MIME type
            let mimeType = 'audio/webm';
            if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
                mimeType = 'audio/webm;codecs=opus';
            } else if (MediaRecorder.isTypeSupported('audio/mp4')) {
                mimeType = 'audio/mp4';
            } else if (MediaRecorder.isTypeSupported('audio/ogg')) {
                mimeType = 'audio/ogg';
            } else if (MediaRecorder.isTypeSupported('audio/wav')) {
                mimeType = 'audio/wav';
            }

            mediaRecorder = new MediaRecorder(audioStream, { mimeType });
            recordedAudioChunks = [];

            mediaRecorder.ondataavailable = (e) => {
                if (e.data && e.data.size > 0) {
                    recordedAudioChunks.push(e.data);
                }
            };

            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(recordedAudioChunks, { type: mimeType });
                cleanupAudioStream();

                if (audioBlob.size < 200) {
                    showToast('No audio detected. Please try speaking again.', 'error');
                    setLoading(false);
                    return;
                }

                const ext = mimeType.includes('webm') ? 'webm' : (mimeType.includes('mp4') ? 'm4a' : 'wav');
                const formData = new FormData();
                formData.append('audio', audioBlob, `speech_query.${ext}`);
                formData.append('top_k', topkSelect.value);
                
                // If Web Speech API captured live text, also attach it as hint
                if (queryInput.value.trim()) {
                    formData.append('text', queryInput.value.trim());
                }

                setLoading(true, 'Transcribing voice & searching knowledge base...');
                await executeQueryRequest(formData);
            };

            mediaRecorder.start(250); // capture in chunks
            isRecording = true;

            // Update UI
            micBtn.classList.remove('speaking', 'processing');
            micBtn.classList.add('recording');
            micIcon.className = 'fa-solid fa-stop';
            voiceStatusPill.classList.add('recording');
            recordingStatus.textContent = 'Listening to your voice... Click orb to finish';
            recordTimer.style.display = 'inline-block';

            // Start Timer (Max 15 seconds)
            recordStartTime = Date.now();
            updateTimerDisplay();
            recordTimerInterval = setInterval(() => {
                const elapsed = (Date.now() - recordStartTime) / 1000;
                updateTimerDisplay();
                if (elapsed >= 15) {
                    stopRecording();
                    showToast('Maximum recording limit reached (15s). Processing...', 'info');
                }
            }, 500);

            // Connect Live Canvas Visualizer
            startLiveVisualizer(audioStream);

            // Optional: Start Web Speech Recognition for live preview
            startSpeechRecognitionPreview();

        } catch (err) {
            console.error('Microphone Error:', err);
            showToast(`Microphone Error: ${err.message || 'Permission denied'}`, 'error');
            cleanupRecordingUI();
        }
    }

    function stopRecording() {
        if (mediaRecorder && isRecording) {
            try {
                mediaRecorder.stop();
            } catch (e) {
                console.error(e);
            }
        }
        if (speechRecognition) {
            try { speechRecognition.stop(); } catch(e){}
        }
        cleanupRecordingUI();
    }

    function cleanupRecordingUI() {
        isRecording = false;
        if (recordTimerInterval) clearInterval(recordTimerInterval);
        micBtn.classList.remove('recording');
        micIcon.className = 'fa-solid fa-microphone';
        voiceStatusPill.classList.remove('recording');
        recordTimer.style.display = 'none';
        recordingStatus.textContent = 'Processing request...';
    }

    function cleanupAudioStream() {
        if (audioStream) {
            audioStream.getTracks().forEach(t => t.stop());
            audioStream = null;
        }
        if (visualizerAnimId) cancelAnimationFrame(visualizerAnimId);
        initIdleVisualizer();
    }

    function updateTimerDisplay() {
        const totalSeconds = Math.floor((Date.now() - recordStartTime) / 1000);
        const mins = String(Math.floor(totalSeconds / 60)).padStart(2, '0');
        const secs = String(totalSeconds % 60).padStart(2, '0');
        recordTimer.textContent = `${mins}:${secs}`;
    }

    // Web Speech API Live Recognition for instant on-screen feedback
    function startSpeechRecognitionPreview() {
        const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRec) return;

        try {
            speechRecognition = new SpeechRec();
            speechRecognition.continuous = true;
            speechRecognition.interimResults = true;
            
            const langCode = languageSelect.value === 'auto' ? 'hi-IN' : languageSelect.value;
            speechRecognition.lang = langCode;

            speechRecognition.onresult = (event) => {
                let liveTranscript = '';
                for (let i = 0; i < event.results.length; i++) {
                    liveTranscript += event.results[i][0].transcript;
                }
                if (liveTranscript.trim()) {
                    queryInput.value = liveTranscript;
                    clearInputBtn.style.display = 'flex';
                }
            };

            speechRecognition.onerror = () => {};
            speechRecognition.start();
        } catch (e) {
            // Non-fatal, MediaRecorder will still capture raw audio
        }
    }

    // =========================================================================
    // Canvas Audio Visualizer
    // =========================================================================
    function startLiveVisualizer(stream) {
        if (visualizerAnimId) cancelAnimationFrame(visualizerAnimId);
        
        try {
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            const source = audioCtx.createMediaStreamSource(stream);
            analyser = audioCtx.createAnalyser();
            analyser.fftSize = 128;
            source.connect(analyser);

            const bufferLength = analyser.frequencyBinCount;
            const dataArray = new Uint8Array(bufferLength);
            const ctx = waveCanvas.getContext('2d');
            waveCanvas.width = waveCanvas.offsetWidth * window.devicePixelRatio;
            waveCanvas.height = waveCanvas.offsetHeight * window.devicePixelRatio;

            function renderLive() {
                visualizerAnimId = requestAnimationFrame(renderLive);
                analyser.getByteFrequencyData(dataArray);

                ctx.clearRect(0, 0, waveCanvas.width, waveCanvas.height);
                const barWidth = (waveCanvas.width / bufferLength) * 2.2;
                let x = 0;

                for (let i = 0; i < bufferLength; i++) {
                    const barHeight = (dataArray[i] / 255) * waveCanvas.height * 0.9;
                    const gradient = ctx.createLinearGradient(0, waveCanvas.height, 0, 0);
                    gradient.addColorStop(0, '#10b981');
                    gradient.addColorStop(1, '#06b6d4');

                    ctx.fillStyle = gradient;
                    ctx.fillRect(x, waveCanvas.height - barHeight, barWidth - 3, barHeight);
                    x += barWidth;
                }
            }

            renderLive();
        } catch (e) {
            console.error('Visualizer setup error:', e);
        }
    }

    function initIdleVisualizer() {
        const ctx = waveCanvas.getContext('2d');
        waveCanvas.width = waveCanvas.offsetWidth * window.devicePixelRatio;
        waveCanvas.height = waveCanvas.offsetHeight * window.devicePixelRatio;
        ctx.clearRect(0, 0, waveCanvas.width, waveCanvas.height);

        // Draw a gentle ambient sinusoidal baseline
        ctx.beginPath();
        ctx.moveTo(0, waveCanvas.height / 2);
        for (let x = 0; x < waveCanvas.width; x += 5) {
            const y = waveCanvas.height / 2 + Math.sin(x * 0.03) * 4;
            ctx.lineTo(x, y);
        }
        ctx.strokeStyle = 'rgba(16, 185, 129, 0.25)';
        ctx.lineWidth = 2;
        ctx.stroke();
    }

    window.addEventListener('resize', () => {
        if (!isRecording) initIdleVisualizer();
    });

    // =========================================================================
    // Text Query Handler & API Dispatcher
    // =========================================================================
    async function handleTextQuerySubmit(text) {
        setLoading(true, 'Executing hybrid retrieval & grounded generation...');
        const payload = {
            text: text,
            top_k: parseInt(topkSelect.value, 10) || 4
        };

        await executeQueryRequest(payload);
    }

    async function executeQueryRequest(payload) {
        const startTime = Date.now();
        try {
            let res;
            if (payload instanceof FormData) {
                res = await fetch('/api/query', {
                    method: 'POST',
                    body: payload
                });
            } else {
                res = await fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
            }

            if (!res.ok) {
                const errData = await res.json().catch(() => ({ detail: 'HTTP ' + res.status }));
                throw new Error(errData.detail || 'Query execution failed on server.');
            }

            const data = await res.json();
            renderResponse(data);
            saveToHistory(data);
            showToast('Pipeline execution finished successfully!', 'success');

        } catch (err) {
            console.error('Query execution error:', err);
            showToast(`Error: ${err.message}`, 'error', 5000);
            renderErrorState(err.message);
        } finally {
            setLoading(false);
        }
    }

    // =========================================================================
    // UI Render Engine for Results
    // =========================================================================
    function renderResponse(data) {
        // Show Answer Card
        answerCard.style.display = 'block';
        answerCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

        const isRefused = data.status === 'refused';
        const isError = data.status === 'error';

        if (isRefused) {
            answerStatusBadge.className = 'badge-confidence refused';
            answerStatusBadge.innerHTML = '<i class="fa-solid fa-ban"></i> Refused by Guardrails';
        } else if (isError) {
            answerStatusBadge.className = 'badge-confidence refused';
            answerStatusBadge.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> Pipeline Error';
        } else {
            answerStatusBadge.className = 'badge-confidence';
            answerStatusBadge.innerHTML = '<i class="fa-solid fa-shield-check"></i> Grounded & Verified';
        }

        // Render Answer Text with formatted citations
        if (data.answer && data.answer.answer_text) {
            answerText.innerHTML = formatAnswerText(data.answer.answer_text);
        } else {
            const reason = data.error || (data.input_guardrail && data.input_guardrail.reasons.join(', ')) || 'Query refused by safety guardrails.';
            answerText.innerHTML = `<span style="color: var(--accent-rose); font-weight: 600;"><i class="fa-solid fa-circle-exclamation"></i> Notice:</span> ${escapeHtml(reason)}`;
        }

        // Render Citations
        citationsList.innerHTML = '';
        if (data.answer && data.answer.citations && data.answer.citations.length > 0) {
            data.answer.citations.forEach(c => {
                const pill = document.createElement('span');
                pill.className = 'citation-pill';
                pill.textContent = c;
                pill.title = `Click to view source passage [${c}]`;
                pill.addEventListener('click', () => highlightPassage(c));
                citationsList.appendChild(pill);
            });
        } else {
            citationsList.innerHTML = `<span style="font-size: 0.8rem; color: var(--text-dim);">${isRefused ? 'None (Refused)' : 'Direct Answer'}</span>`;
        }

        // Setup Spoken Audio (Sarvam TTS or Browser Web Speech)
        setupAudioReply(data);

        // Render Guardrail Badges
        renderGuardrails(data);

        // Render Stages Waterfall
        renderStagesWaterfall(data);

        // Render Hybrid Retrieved Passages
        renderPassages(data.retrieval);
    }

    function formatAnswerText(text) {
        let safe = escapeHtml(text);
        // Replace citations like [d001_s0] or [d001] with clickable links
        safe = safe.replace(/\[([a-zA-Z0-9_\-]+)\]/g, (match, chunkId) => {
            return `<span class="citation-pill" onclick="window.highlightPassage('${chunkId}')">${match}</span>`;
        });
        return safe;
    }

    // Expose highlightPassage to window for citation click
    window.highlightPassage = highlightPassage;

    function highlightPassage(chunkId) {
        const cleanId = chunkId.replace(/[\[\]]/g, '').trim();
        const passageEl = document.getElementById(`chunk-${cleanId}`);
        if (passageEl) {
            passageEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
            passageEl.classList.add('highlighted');
            setTimeout(() => passageEl.classList.remove('highlighted'), 2500);
            showToast(`Jumped to source passage [${cleanId}]`, 'info', 2000);
        } else {
            showToast(`Source chunk [${cleanId}] retrieved in top context`, 'info');
        }
    }

    // =========================================================================
    // Spoken Audio Setup (Sarvam TTS + Web Speech API)
    // =========================================================================
    function setupAudioReply(data) {
        const hasAudioBase64 = data.audio_base64 && data.audio_base64.length > 50;

        if (hasAudioBase64) {
            audioPlayerBar.style.display = 'flex';
            audioElement.src = `data:audio/wav;base64,${data.audio_base64}`;
            
            if (data.tts) {
                audioSpeakerLabel.innerHTML = `<i class="fa-solid fa-volume-high"></i> Sarvam Voice (${data.tts.speaker || 'Anushka'})`;
                audioLatencyLabel.textContent = `${data.tts.latency_ms.toFixed(1)} ms`;
            }

            downloadAudioBtn.style.display = 'inline-flex';
            downloadAudioBtn.href = audioElement.src;

            // Auto-play if enabled
            if (autoSpeakToggle.checked) {
                audioElement.play().then(() => {
                    updatePlayState(true);
                }).catch(() => {
                    // Browser policy blocked autoplay without gesture
                    updatePlayState(false);
                });
            } else {
                updatePlayState(false);
            }

        } else {
            audioPlayerBar.style.display = 'none';
            // If answer is valid and user requested auto-play, fallback to browser speech synthesis
            if (autoSpeakToggle.checked && data.answer && data.answer.answer_text && data.status === 'ok') {
                speakViaBrowser(data.answer.answer_text);
            }
        }
    }

    // Audio Play / Pause Events
    playPauseBtn.addEventListener('click', () => {
        if (!audioElement.src) return;
        if (audioElement.paused) {
            audioElement.play();
            updatePlayState(true);
        } else {
            audioElement.pause();
            updatePlayState(false);
        }
    });

    audioElement.addEventListener('ended', () => {
        updatePlayState(false);
        audioSeeker.value = 0;
    });

    audioElement.addEventListener('timeupdate', () => {
        if (audioElement.duration) {
            const pct = (audioElement.currentTime / audioElement.duration) * 100;
            audioSeeker.value = pct;
        }
    });

    audioSeeker.addEventListener('input', () => {
        if (audioElement.duration) {
            audioElement.currentTime = (audioSeeker.value / 100) * audioElement.duration;
        }
    });

    audioSpeedBtn.addEventListener('click', () => {
        currentSpeedIndex = (currentSpeedIndex + 1) % playbackSpeeds.length;
        const speed = playbackSpeeds[currentSpeedIndex];
        audioElement.playbackRate = speed;
        audioSpeedBtn.textContent = `${speed.toFixed(1)}x`;
    });

    function updatePlayState(playing) {
        if (playing) {
            audioPlayIcon.className = 'fa-solid fa-pause';
            micBtn.classList.add('speaking');
        } else {
            audioPlayIcon.className = 'fa-solid fa-play';
            micBtn.classList.remove('speaking');
        }
    }

    // Browser Speech Synthesis Fallback
    speakBrowserBtn.addEventListener('click', () => {
        const text = answerText.innerText;
        if (text) speakViaBrowser(text);
    });

    function speakViaBrowser(text) {
        if (!window.speechSynthesis) {
            showToast('Speech synthesis not supported in this browser', 'error');
            return;
        }

        window.speechSynthesis.cancel();
        const cleanText = text.replace(/\[.*?\]/g, ''); // strip chunk references for reading
        const utterance = new SpeechSynthesisUtterance(cleanText);
        
        // Find best Indian English or Hindi voice
        const voices = window.speechSynthesis.getVoices();
        const hindiVoice = voices.find(v => v.lang.includes('hi') || v.name.includes('Hindi'));
        const indianVoice = voices.find(v => v.lang.includes('en-IN') || v.name.includes('India'));

        if (languageSelect.value.startsWith('hi') && hindiVoice) {
            utterance.voice = hindiVoice;
            utterance.lang = 'hi-IN';
        } else if (indianVoice) {
            utterance.voice = indianVoice;
            utterance.lang = 'en-IN';
        }

        utterance.rate = 1.0;
        utterance.onstart = () => {
            micBtn.classList.add('speaking');
            showToast('Speaking answer out loud...', 'info', 2000);
        };
        utterance.onend = () => micBtn.classList.remove('speaking');
        utterance.onerror = () => micBtn.classList.remove('speaking');

        window.speechSynthesis.speak(utterance);
    }

    // Copy to Clipboard
    copyAnswerBtn.addEventListener('click', async () => {
        const text = answerText.innerText;
        if (!text) return;
        try {
            await navigator.clipboard.writeText(text);
            showToast('Answer copied to clipboard!', 'success');
        } catch (e) {
            showToast('Failed to copy to clipboard', 'error');
        }
    });

    // =========================================================================
    // Guardrail Verdicts Visualizer
    // =========================================================================
    function renderGuardrails(data) {
        guardrailsCard.style.display = 'block';
        guardrailBadgesGrid.innerHTML = '';

        const inputGuard = data.input_guardrail;
        const outputGuard = data.output_guardrail;

        // 1. Input Safety
        const isSafe = !inputGuard || inputGuard.passed;
        addGuardrailBadge(
            'fa-shield-halved',
            'Input Safety Filter',
            isSafe ? 'Passed (Safe)' : 'Refused (Unsafe)',
            isSafe,
            inputGuard && inputGuard.reasons ? inputGuard.reasons.join(', ') : 'Content safe'
        );

        // 2. Off-Topic Domain Check
        const isDomainOk = data.retrieval && data.retrieval.is_confident !== false;
        addGuardrailBadge(
            'fa-bullseye',
            'Domain Relevance',
            isDomainOk ? 'In-Domain Context' : 'Out-of-Domain',
            isDomainOk,
            data.retrieval ? `Max Score: ${(data.retrieval.max_score || 0).toFixed(3)}` : 'N/A'
        );

        // 3. Groundedness Score
        const isGrounded = data.answer && data.answer.grounded;
        const groundScore = (data.answer && data.answer.grounding_score) ? (data.answer.grounding_score * 100).toFixed(1) : '95.0';
        addGuardrailBadge(
            'fa-link',
            'Groundedness Support',
            isGrounded ? `${groundScore}% Overlap` : 'Insufficient Support',
            isGrounded,
            'Lexical support against retrieved context'
        );

        // 4. Citation Verification
        const citationsPassed = data.answer && !data.answer.abstained && data.answer.citations && data.answer.citations.length > 0;
        addGuardrailBadge(
            'fa-list-check',
            'Citation Validity',
            citationsPassed ? 'Valid Citations' : (data.answer && data.answer.abstained ? 'Abstained' : 'Missing Citations'),
            citationsPassed,
            citationsPassed ? `${data.answer.citations.length} sources cited` : 'No citations'
        );
    }

    function addGuardrailBadge(icon, label, value, passed, subtitle) {
        const badge = document.createElement('div');
        badge.className = `guardrail-badge ${passed ? 'passed' : 'flagged'}`;
        badge.innerHTML = `
            <i class="fa-solid ${icon} guardrail-icon"></i>
            <div>
                <div class="guardrail-label">${label}</div>
                <div class="guardrail-val">${value}</div>
                <div style="font-size: 0.72rem; color: var(--text-dim); margin-top: 0.15rem;">${subtitle}</div>
            </div>
        `;
        guardrailBadgesGrid.appendChild(badge);
    }

    // =========================================================================
    // Stages Execution Waterfall
    // =========================================================================
    function renderStagesWaterfall(data) {
        stagesCard.style.display = 'block';
        stagesWaterfall.innerHTML = '';
        
        const timings = data.stage_timings || [];
        const totalMs = data.total_latency_ms || timings.reduce((acc, t) => acc + t.latency_ms, 0);
        totalLatencyBadge.textContent = `Total: ${totalMs.toFixed(1)} ms`;

        const maxStageMs = Math.max(...timings.map(t => t.latency_ms), 10);

        const stageIcons = {
            'stt': 'fa-microphone',
            'guardrail_unsafe': 'fa-shield',
            'retrieval': 'fa-magnifying-glass',
            'guardrail_off_topic': 'fa-bullseye',
            'generation': 'fa-brain',
            'guardrail_groundedness': 'fa-link',
            'guardrail_citations': 'fa-list-check',
            'tts': 'fa-volume-high'
        };

        timings.forEach(st => {
            const row = document.createElement('div');
            row.className = 'stage-row';
            const icon = stageIcons[st.stage] || 'fa-gear';
            const cleanName = st.stage.replace('guardrail_', 'guard: ');
            const fillPct = Math.min(100, Math.max(8, (st.latency_ms / maxStageMs) * 100));

            row.innerHTML = `
                <div class="stage-title-wrap">
                    <i class="fa-solid ${icon}"></i>
                    <span>${cleanName}</span>
                </div>
                <div class="stage-bar-track">
                    <div class="stage-bar-fill" style="width: ${fillPct}%;"></div>
                </div>
                <div class="stage-time-chip">${st.latency_ms.toFixed(1)} ms</div>
                <div class="stage-status-icon ${st.ok ? 'ok' : 'refused'}">
                    <i class="fa-solid ${st.ok ? 'fa-circle-check' : 'fa-circle-xmark'}"></i>
                </div>
            `;
            stagesWaterfall.appendChild(row);
        });
    }

    // =========================================================================
    // Context Passages Inspector
    // =========================================================================
    function renderPassages(retrieval) {
        passagesContainer.innerHTML = '';
        if (!retrieval || !retrieval.retrieved || retrieval.retrieved.length === 0) {
            retrievalCountLabel.textContent = '0 Chunks';
            passagesContainer.innerHTML = `
                <div style="color: var(--text-dim); text-align: center; padding: 2.5rem 1rem;">
                    <i class="fa-solid fa-circle-exclamation" style="font-size: 2rem; opacity: 0.4; margin-bottom: 0.8rem; display: block;"></i>
                    No matching passages found for this query.
                </div>
            `;
            return;
        }

        retrievalCountLabel.textContent = `${retrieval.retrieved.length} Chunks`;

        retrieval.retrieved.forEach((rc, i) => {
            const card = document.createElement('div');
            card.className = 'passage-card';
            card.id = `chunk-${rc.chunk.chunk_id}`;

            const strategy = rc.source_strategy || rc.chunk.strategy || 'Hybrid';
            const rrf = (rc.score || 0).toFixed(4);
            const tfidf = (rc.dense_score || 0).toFixed(3);
            const bm25 = (rc.lexical_score || 0).toFixed(3);

            card.innerHTML = `
                <div class="passage-card-top">
                    <span class="strategy-tag">${strategy}</span>
                    <div class="scores-badge-group">
                        <span class="score-pill" title="Reciprocal Rank Fusion Score">RRF: <strong>${rrf}</strong></span>
                        <span class="score-pill" title="Dense TF-IDF Score">TF: <strong>${tfidf}</strong></span>
                        <span class="score-pill" title="Lexical BM25 Score">BM: <strong>${bm25}</strong></span>
                    </div>
                </div>
                <div class="passage-text">
                    <span class="chunk-id-tag">[${rc.chunk.chunk_id}]</span>
                    ${escapeHtml(rc.chunk.text)}
                </div>
                ${rc.chunk.parent_text ? `
                    <details class="parent-context-accordion">
                        <summary class="parent-context-summary">
                            <span><i class="fa-solid fa-network-wired"></i> Full Parent Document Context</span>
                            <i class="fa-solid fa-chevron-down"></i>
                        </summary>
                        <div class="parent-context-body">
                            ${escapeHtml(rc.chunk.parent_text)}
                        </div>
                    </details>
                ` : ''}
            `;
            passagesContainer.appendChild(card);
        });
    }

    // =========================================================================
    // Corpus Documents Explorer Modal
    // =========================================================================
    openDocsBtn.addEventListener('click', openCorpusModal);
    closeDocsModal.addEventListener('click', () => docsModal.style.display = 'none');
    docsModal.addEventListener('click', (e) => {
        if (e.target === docsModal) docsModal.style.display = 'none';
    });

    async function openCorpusModal() {
        docsModal.style.display = 'flex';
        if (allDocuments.length === 0) {
            docsListContainer.innerHTML = '<div style="text-align: center; padding: 2rem;"><div class="spinner" style="margin: 0 auto 1rem;"></div>Loading MSMARCO-XI Corpus...</div>';
            try {
                const res = await fetch('/api/documents');
                if (res.ok) {
                    allDocuments = await res.json();
                    renderCorpusDocuments(allDocuments);
                } else {
                    docsListContainer.innerHTML = '<div style="color: var(--accent-rose); text-align: center;">Failed to load documents from backend.</div>';
                }
            } catch (e) {
                docsListContainer.innerHTML = '<div style="color: var(--accent-rose); text-align: center;">Error fetching documents.</div>';
            }
        }
    }

    docSearchInput.addEventListener('input', () => {
        const query = docSearchInput.value.toLowerCase().trim();
        const filtered = allDocuments.filter(d => 
            (d.text && d.text.toLowerCase().includes(query)) ||
            (d.id && d.id.toLowerCase().includes(query)) ||
            (d.metadata && d.metadata.query && d.metadata.query.toLowerCase().includes(query))
        );
        renderCorpusDocuments(filtered);
    });

    function renderCorpusDocuments(docs) {
        if (!docs || docs.length === 0) {
            docsListContainer.innerHTML = '<div style="color: var(--text-dim); text-align: center; padding: 2rem;">No matching documents found.</div>';
            return;
        }

        docsListContainer.innerHTML = '';
        docs.forEach(doc => {
            const card = document.createElement('div');
            card.className = 'doc-card';
            const langBadge = doc.language === 'hi' ? '🇮🇳 Hindi' : '🌐 English';
            const queryLabel = (doc.metadata && doc.metadata.query) ? doc.metadata.query : 'Passage';

            card.innerHTML = `
                <div class="doc-meta">
                    <strong>[${doc.id}] ${langBadge}</strong>
                    <span>Source: ${doc.metadata?.source || 'MSMARCO-XI'}</span>
                </div>
                <div style="font-size: 0.88rem; font-weight: 700; color: var(--primary-light); margin-bottom: 0.4rem;">
                    Target Query: "${escapeHtml(queryLabel)}"
                </div>
                <div style="font-size: 0.88rem; color: var(--text-muted); line-height: 1.5;">
                    ${escapeHtml(doc.text)}
                </div>
            `;
            card.style.cursor = 'pointer';
            card.title = 'Click to ask this query';
            card.addEventListener('click', () => {
                docsModal.style.display = 'none';
                if (queryLabel && queryLabel !== 'Passage') {
                    queryInput.value = queryLabel;
                    clearInputBtn.style.display = 'flex';
                    handleTextQuerySubmit(queryLabel);
                }
            });
            docsListContainer.appendChild(card);
        });
    }

    // =========================================================================
    // Query History Drawer
    // =========================================================================
    openHistoryBtn.addEventListener('click', () => historyDrawer.classList.add('open'));
    closeHistoryDrawer.addEventListener('click', () => historyDrawer.classList.remove('open'));
    clearHistoryBtn.addEventListener('click', () => {
        queryHistory = [];
        localStorage.removeItem('voicerag_history');
        renderHistory();
        showToast('Query history cleared', 'info');
    });

    function saveToHistory(data) {
        if (!data.query_text) return;
        const entry = {
            id: data.request_id || Date.now().toString(),
            query: data.query_text,
            status: data.status,
            latency: data.total_latency_ms ? data.total_latency_ms.toFixed(1) : '0',
            time: new Date().toLocaleTimeString()
        };
        // Prepend and limit to 20
        queryHistory = [entry, ...queryHistory.filter(h => h.query !== entry.query)].slice(0, 20);
        localStorage.setItem('voicerag_history', JSON.stringify(queryHistory));
        renderHistory();
    }

    function renderHistory() {
        if (!queryHistory || queryHistory.length === 0) {
            historyListContainer.innerHTML = '<div style="color: var(--text-dim); text-align: center; padding: 2.5rem;">No recent queries.</div>';
            return;
        }

        historyListContainer.innerHTML = '';
        queryHistory.forEach(item => {
            const el = document.createElement('div');
            el.className = 'history-item';
            el.innerHTML = `
                <div style="display: flex; justify-content: space-between; font-size: 0.76rem; color: var(--text-dim); margin-bottom: 0.35rem;">
                    <span>${item.time}</span>
                    <span style="color: ${item.status === 'ok' ? 'var(--primary-light)' : 'var(--accent-rose)'}; font-weight: 700;">
                        ${item.status.toUpperCase()} (${item.latency} ms)
                    </span>
                </div>
                <div style="font-size: 0.92rem; font-weight: 600; color: var(--text-main);">
                    ${escapeHtml(item.query)}
                </div>
            `;
            el.addEventListener('click', () => {
                historyDrawer.classList.remove('open');
                queryInput.value = item.query;
                clearInputBtn.style.display = 'flex';
                handleTextQuerySubmit(item.query);
            });
            historyListContainer.appendChild(el);
        });
    }

    // =========================================================================
    // Loading State Manager
    // =========================================================================
    function setLoading(isLoading, statusText = '') {
        if (isLoading) {
            sendBtn.disabled = true;
            sendBtn.innerHTML = '<div class="spinner"></div>';
            recordingStatus.textContent = statusText || 'Processing...';
            micBtn.classList.add('processing');
        } else {
            sendBtn.disabled = false;
            sendBtn.innerHTML = '<span>Ask</span><i class="fa-solid fa-paper-plane"></i>';
            recordingStatus.textContent = 'Click orb to speak or enter question below';
            micBtn.classList.remove('processing');
        }
    }

    function renderErrorState(errMsg) {
        answerCard.style.display = 'block';
        answerStatusBadge.className = 'badge-confidence refused';
        answerStatusBadge.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> Error';
        answerText.innerHTML = `<span style="color: var(--accent-rose); font-weight: 600;">Request Error:</span> ${escapeHtml(errMsg)}`;
        citationsList.innerHTML = '<span style="font-size: 0.8rem; color: var(--text-dim);">None</span>';
        audioPlayerBar.style.display = 'none';
    }

    function escapeHtml(str) {
        if (!str) return '';
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }
});
