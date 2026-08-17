/* Natural Theme JavaScript App for Voice RAG */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const micBtn = document.getElementById('mic-btn');
    const micIcon = document.getElementById('mic-icon');
    const recordingStatus = document.getElementById('recording-status');
    const queryInput = document.getElementById('query-input');
    const sendBtn = document.getElementById('send-btn');
    const presets = document.querySelectorAll('.preset-pill');
    
    const answerCard = document.getElementById('answer-card');
    const answerText = document.getElementById('answer-text');
    const citationsList = document.getElementById('citations-list');
    
    const audioContainer = document.getElementById('audio-container');
    const playAudioBtn = document.getElementById('play-audio-btn');
    const playIcon = document.getElementById('play-icon');
    const audioElement = document.getElementById('audio-element');
    const ttsLatencyInfo = document.getElementById('tts-latency-info');
    
    const timingCard = document.getElementById('timing-card');
    const stagesGrid = document.getElementById('stages-grid');
    const passagesContainer = document.getElementById('passages-container');
    
    const backendInfo = document.getElementById('backend-info');

    // MediaRecorder & AudioContext variables
    let mediaRecorder = null;
    let audioChunks = [];
    let isRecording = false;
    let audioCtx = null;
    let analyser = null;
    let animId = null;

    // Initial Health Check
    fetchHealth();

    async function fetchHealth() {
        try {
            const res = await fetch('/api/health');
            if (res.ok) {
                const data = await res.json();
                backendInfo.textContent = `Online • ${data.generator_backend} (${data.docs_count} docs)`;
            } else {
                backendInfo.textContent = 'Backend Offline';
            }
        } catch (e) {
            backendInfo.textContent = 'Connection Error';
        }
    }

    // Preset Pill Clicks
    presets.forEach(pill => {
        pill.addEventListener('click', () => {
            queryInput.value = pill.getAttribute('data-query');
            submitTextQuery();
        });
    });

    // Send Button Click
    sendBtn.addEventListener('click', () => submitTextQuery());

    queryInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            submitTextQuery();
        }
    });

    // Submit Text Query
    function submitTextQuery() {
        const text = queryInput.value.trim();
        if (!text) return;

        setLoading(true);
        const formData = new FormData();
        formData.append('text', text);

        sendApiQuery(formData);
    }

    // Microphone Recording Logic
    micBtn.addEventListener('click', toggleRecording);

    async function toggleRecording() {
        if (!isRecording) {
            startRecording();
        } else {
            stopRecording();
        }
    }

    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];

            mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) {
                    audioChunks.push(e.data);
                }
            };

            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                const formData = new FormData();
                formData.append('audio', audioBlob, 'recording.wav');
                setLoading(true);
                sendApiQuery(formData);
            };

            mediaRecorder.start();
            isRecording = true;
            micBtn.classList.add('recording');
            micIcon.className = 'fa-solid fa-stop';
            recordingStatus.textContent = '🔴 Listening to your voice... Click to stop!';

            // Setup Visualizer Canvas
            setupVisualizer(stream);

        } catch (err) {
            alert('Microphone permission error: ' + err.message);
        }
    }

    function stopRecording() {
        if (mediaRecorder && isRecording) {
            mediaRecorder.stop();
            isRecording = false;
            micBtn.classList.remove('recording');
            micIcon.className = 'fa-solid fa-microphone';
            recordingStatus.textContent = 'Processing voice query...';
            if (animId) cancelAnimationFrame(animId);
        }
    }

    // Audio Visualizer Canvas Animation
    function setupVisualizer(stream) {
        const canvas = document.getElementById('wave-canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;

        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const source = audioCtx.createMediaStreamSource(stream);
        analyser = audioCtx.createAnalyser();
        analyser.fftSize = 64;
        source.connect(analyser);

        const dataArray = new Uint8Array(analyser.frequencyBinCount);

        function draw() {
            animId = requestAnimationFrame(draw);
            analyser.getByteFrequencyData(dataArray);

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = 'rgba(82, 183, 136, 0.15)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            const barWidth = (canvas.width / dataArray.length) * 1.5;
            let x = 0;

            for (let i = 0; i < dataArray.length; i++) {
                const barHeight = (dataArray[i] / 255) * canvas.height;
                ctx.fillStyle = '#2d6a4f';
                ctx.fillRect(x, canvas.height - barHeight, barWidth - 2, barHeight);
                x += barWidth;
            }
        }

        draw();
    }

    // Send FormData to /api/query
    async function sendApiQuery(formData) {
        try {
            const res = await fetch('/api/query', {
                method: 'POST',
                body: formData
            });

            if (!res.ok) throw new Error('API Request Failed');
            const data = await res.json();
            renderResults(data);

        } catch (err) {
            alert('Error: ' + err.message);
        } finally {
            setLoading(false);
        }
    }

    // Render Response Data into UI
    function renderResults(data) {
        // Display Answer Box
        answerCard.style.display = 'block';
        if (data.answer) {
            answerText.textContent = data.answer.answer_text;
            
            // Citations
            citationsList.innerHTML = '';
            if (data.answer.citations && data.answer.citations.length > 0) {
                data.answer.citations.forEach(c => {
                    const tag = document.createElement('span');
                    tag.className = 'citation-tag';
                    tag.textContent = c;
                    citationsList.appendChild(tag);
                });
            } else {
                citationsList.innerHTML = '<span style="font-size: 0.8rem; color: var(--text-muted);">None</span>';
            }
        } else {
            answerText.textContent = data.error || 'Query refused by pipeline guardrails.';
            citationsList.innerHTML = '';
        }

        // Spoken Audio Output Player
        if (data.audio_base64) {
            audioContainer.style.display = 'flex';
            audioElement.src = 'data:audio/wav;base64,' + data.audio_base64;
            audioElement.play();
            playIcon.className = 'fa-solid fa-pause';
            if (data.tts) {
                ttsLatencyInfo.textContent = `Synthesized in ${data.tts.latency_ms.toFixed(1)} ms (${data.tts.speaker} voice)`;
            }
        } else {
            audioContainer.style.display = 'none';
        }

        // Audio Play/Pause Button
        playAudioBtn.onclick = () => {
            if (audioElement.paused) {
                audioElement.play();
                playIcon.className = 'fa-solid fa-pause';
            } else {
                audioElement.pause();
                playIcon.className = 'fa-solid fa-play';
            }
        };

        audioElement.onended = () => {
            playIcon.className = 'fa-solid fa-play';
        };

        // Render Stage Timings
        timingCard.style.display = 'block';
        stagesGrid.innerHTML = '';
        if (data.stage_timings) {
            data.stage_timings.forEach(st => {
                const step = document.createElement('div');
                step.className = `stage-step ${st.ok ? 'ok' : 'refused'}`;
                step.innerHTML = `
                    <div class="stage-name">${st.stage.replace('guardrail_', '')}</div>
                    <div class="stage-time">${st.latency_ms.toFixed(1)} ms</div>
                `;
                stagesGrid.appendChild(step);
            });
        }

        // Render Retrieved Passages
        passagesContainer.innerHTML = '';
        if (data.retrieval && data.retrieval.retrieved && data.retrieval.retrieved.length > 0) {
            data.retrieval.retrieved.forEach((rc, i) => {
                const card = document.createElement('div');
                card.className = 'passage-card';
                card.innerHTML = `
                    <div class="passage-header">
                        <span class="strategy-badge">${rc.source_strategy || rc.chunk.strategy}</span>
                        <div class="scores-group">
                            <span>RRF: ${rc.score.toFixed(3)}</span>
                            <span>TF-IDF: ${rc.dense_score.toFixed(2)}</span>
                            <span>BM25: ${rc.lexical_score.toFixed(2)}</span>
                        </div>
                    </div>
                    <div class="passage-body">
                        <strong>[${rc.chunk.chunk_id}]</strong> ${rc.chunk.text}
                    </div>
                    ${rc.chunk.parent_text ? `
                        <div class="parent-context">
                            <strong>Parent Passage:</strong> ${rc.chunk.parent_text}
                        </div>
                    ` : ''}
                `;
                passagesContainer.appendChild(card);
            });
        } else {
            passagesContainer.innerHTML = '<div style="color: var(--text-muted); text-align: center; padding: 2rem 0;">No passages retrieved for this query.</div>';
        }
    }

    function setLoading(loading) {
        if (loading) {
            sendBtn.disabled = true;
            sendBtn.innerHTML = '<div class="spinner"></div>';
            recordingStatus.textContent = 'Processing...';
        } else {
            sendBtn.disabled = false;
            sendBtn.innerHTML = '<i class="fa-solid fa-paper-plane"></i> Ask';
            recordingStatus.textContent = 'Click microphone or type query';
        }
    }
});
