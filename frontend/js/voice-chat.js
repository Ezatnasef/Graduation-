/**
 * Servia Voice Chat - Egyptian Dialect TTS + VAD System
 * Features:
 * - 4 Egyptian dialect TTS (Cairene, Sa'idi, Alexandrian, Bedouin)
 * - Real-time Voice Activity Detection (VAD)
 * - Barge-in (interruption) support
 * - Continuous listening mode
 * - WebSocket real-time communication
 */

(function () {
    'use strict';

    // HARDCODED backend connection (always localhost:8765)
    // Do NOT depend on window.location since frontend might be opened via file:// or different port
    // This ensures consistent backend connection regardless of how frontend is hosted
    const BACKEND_HOST = 'localhost:8765';
    const BACKEND_HTTPS = false; // Set to true if backend is on HTTPS
    
    const runtimeHost = BACKEND_HOST;
    const runtimeHttpScheme = BACKEND_HTTPS ? 'https:' : 'http:';
    const runtimeWsScheme = BACKEND_HTTPS ? 'wss:' : 'ws:';

    // ===================== CONFIG =====================
    const CONFIG = {
        wsUrl: `${runtimeWsScheme}//${runtimeHost}/ws/voice`,
        apiUrl: `${runtimeHttpScheme}//${runtimeHost}`,
        sampleRate: 16000,
        vadFrameSize: 480,         // 30ms at 16kHz
        vadEnergyThreshold: 0.012,
        vadSilenceMs: 1200,
        vadSpeechMinMs: 250,
        bargeInThreshold: 0.035,
        bargeInFrames: 14,
        visualizerBars: 40,
        recognitionDedupMs: 2500,
        recognitionSuppressAfterTTSMs: 3500,
        recognitionEchoBlockMs: 18000,
    };

    // ===================== STATE =====================
    const STATE = {
        // Connection
        ws: null,
        connected: false,
        reconnectTimer: null,
        reconnectAttempts: 0,

        // Dialect
        dialect: 'cairene',
        gender: 'female',

        // Recording
        isListening: false,
        continuousMode: false,
        stream: null,
        audioContext: null,
        processor: null,
        audioInput: null,

        // VAD
        isSpeaking: false,
        silenceFrames: 0,
        speechFrames: 0,
        audioBuffer: [],
        bufferLength: 0,

        // TTS
        isTTSPlaying: false,
        currentAudio: null,
        ttsQueue: [],
        ttsServerDone: true,
        lastBotMessageEl: null,
        activeResponseId: null,
        lastBotNormalizedText: '',
        lastTTSFinishedAt: 0,

        // Speech Recognition (fallback)
        recognition: null,
        isRecognizing: false,
        suppressRecognitionUntil: 0,
        lastSentTranscript: '',
        lastSentAt: 0,

        // UI
        vadBars: [],
    };

    // ===================== DOM =====================
    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    const DOM = {
        messages: $('#messagesContainer'),
        textInput: $('#textInput'),
        sendBtn: $('#sendBtn'),
        micBtn: $('#micBtn'),
        micLabel: $('#micLabel'),
        continuousToggle: $('#continuousToggle'),
        dialectBtns: $$('.dialect-btn'),
        genderToggle: $('#genderToggle'),
        statusDot: $('#statusDot'),
        statusText: $('#statusText'),
        vadVisualizer: $('#vadVisualizer'),
        vadBars: $('#vadBars'),
        vadStatusLabel: $('#vadStatusLabel'),
        vadEnergyFill: $('#vadEnergyFill'),
        interruptBtn: $('#interruptBtn'),
        clearBtn: $('#clearBtn'),
        sentimentList: $('#sentimentList'),
        sentimentEmpty: $('#sentimentEmpty'),
        sentimentCurrent: $('#sentimentCurrent'),
        sessionInfo: $('#sessionInfo'),
    };

    // ===================== HELPERS =====================
    function formatTime() {
        return new Date().toLocaleTimeString('ar-EG', {
            hour: '2-digit', minute: '2-digit'
        });
    }

    function scrollToBottom() {
        if (DOM.messages) {
            DOM.messages.scrollTop = DOM.messages.scrollHeight;
        }
    }

    const DIALECT_NAMES = {
        cairene: 'قاهرية',
        saidi: 'صعيدية',
        alexandrian: 'إسكندرانية',
        bedouin: 'بدوية',
    };

    function normalizeAudioFormat(format) {
        const value = (format || 'mp3').toString().toLowerCase();
        if (value === 'wav' || value === 'wave') return 'wav';
        if (value === 'mp3' || value === 'mpeg') return 'mp3';
        return 'mp3';
    }

    function normalizeTranscript(text) {
        return (text || '')
            .replace(/\s+/g, ' ')
            .trim()
            .toLowerCase();
    }

    function isLikelyBotEcho(normalizedTranscript) {
        if (!normalizedTranscript || !STATE.lastBotNormalizedText) {
            return false;
        }

        const now = Date.now();
        if (now - STATE.lastTTSFinishedAt > CONFIG.recognitionEchoBlockMs) {
            return false;
        }

        if (normalizedTranscript === STATE.lastBotNormalizedText) {
            return true;
        }

        if (normalizedTranscript.length < 6) {
            return false;
        }

        return (
            STATE.lastBotNormalizedText.includes(normalizedTranscript) ||
            normalizedTranscript.includes(STATE.lastBotNormalizedText)
        );
    }

    // ===================== MESSAGES =====================
    function addMessage(text, type = 'bot', options = {}) {
        const msg = document.createElement('div');
        msg.className = `message ${type}`;

        // Dialect badge for bot messages
        if (type === 'bot' && options.dialect) {
            const badge = document.createElement('div');
            badge.className = 'dialect-badge';
            badge.textContent = DIALECT_NAMES[options.dialect] || options.dialect;
            msg.appendChild(badge);
        }

        const content = document.createElement('div');
        content.textContent = text;
        msg.appendChild(content);

        // Audio player for TTS
        if (options.audioBase64) {
            const audio = document.createElement('audio');
            audio.className = 'audio-player';
            audio.controls = true;
            const audioFormat = normalizeAudioFormat(options.audioFormat || 'mp3');
            audio.src = `data:audio/${audioFormat};base64,${options.audioBase64}`;
            msg.appendChild(audio);
        }

        const ts = document.createElement('div');
        ts.className = 'timestamp';
        ts.textContent = formatTime();
        msg.appendChild(ts);

        // Click bot messages to replay TTS
        if (type === 'bot') {
            msg.title = 'اضغط لإعادة تشغيل الصوت';
            msg.addEventListener('click', (event) => {
                if (event.target && event.target.closest('audio')) return;
                replayBotMessage(msg, text);
            });
        }

        DOM.messages.appendChild(msg);
        scrollToBottom();
        return msg;
    }

    function addThinking() {
        const msg = document.createElement('div');
        msg.className = 'message bot thinking-msg';
        msg.innerHTML = '<div class="thinking-dots"><span></span><span></span><span></span></div>';
        DOM.messages.appendChild(msg);
        scrollToBottom();
        return msg;
    }

    function mapSentimentLabel(label) {
        const key = (label || 'neutral').toLowerCase();
        const map = {
            happy: { text: 'مبسوط/فرحان', cls: 'sentiment-happy' },
            excited: { text: 'متحمس', cls: 'sentiment-happy' },
            angry: { text: 'زعلان/غاضب', cls: 'sentiment-angry' },
            sad: { text: 'حزين', cls: 'sentiment-sad' },
            frustrated: { text: 'متضايق', cls: 'sentiment-frustrated' },
            concerned: { text: 'قلقان', cls: 'sentiment-concerned' },
            neutral: { text: 'محايد', cls: 'sentiment-neutral' },
        };
        return map[key] || map.neutral;
    }

    function mapIntentLabel(label) {
        const map = {
            complaint: 'شكوى',
            inquiry: 'استفسار',
            request: 'طلب',
            cancellation_or_refund: 'إلغاء/استرجاع',
            technical_issue: 'مشكلة تقنية',
            praise: 'إشادة',
            greeting: 'ترحيب',
            feedback: 'ملاحظات',
            other: 'عام',
        };
        return map[(label || 'other').toLowerCase()] || 'عام';
    }

    function mapUrgencyLabel(label) {
        const map = {
            low: 'منخفض',
            medium: 'متوسط',
            high: 'عالي',
        };
        return map[(label || 'low').toLowerCase()] || 'منخفض';
    }

    function updateCurrentSentimentPill(sentimentLabel) {
        if (!DOM.sentimentCurrent) return;
        const info = mapSentimentLabel(sentimentLabel);
        DOM.sentimentCurrent.className = `sentiment-current ${info.cls}`;
        DOM.sentimentCurrent.textContent = info.text;
    }

    function addSentimentEntry(userText, analysis) {
        if (!DOM.sentimentList || !analysis) return;

        if (DOM.sentimentEmpty) {
            DOM.sentimentEmpty.style.display = 'none';
        }

        const sentiment = mapSentimentLabel(analysis.sentiment_label);
        updateCurrentSentimentPill(analysis.sentiment_label);

        const item = document.createElement('div');
        item.className = 'sentiment-item';

        const head = document.createElement('div');
        head.className = 'sentiment-item-head';

        const badge = document.createElement('span');
        badge.className = `sentiment-badge ${sentiment.cls}`;
        badge.textContent = sentiment.text;

        const time = document.createElement('span');
        time.className = 'sentiment-time';
        time.textContent = formatTime();

        head.appendChild(badge);
        head.appendChild(time);

        const msg = document.createElement('div');
        msg.className = 'sentiment-message';
        msg.textContent = (userText || '').slice(0, 120) || 'رسالة بدون نص';

        const meta = document.createElement('div');
        meta.className = 'sentiment-meta';

        const intent = document.createElement('span');
        intent.textContent = `النية: ${mapIntentLabel(analysis.intent_label)}`;

        const urgency = document.createElement('span');
        urgency.textContent = `الاستعجال: ${mapUrgencyLabel(analysis.urgency)}`;

        meta.appendChild(intent);
        meta.appendChild(urgency);

        if (analysis.needs_human_agent) {
            const escalation = document.createElement('span');
            escalation.textContent = 'يحتاج تصعيد';
            meta.appendChild(escalation);
        }

        item.appendChild(head);
        item.appendChild(msg);
        item.appendChild(meta);

        DOM.sentimentList.prepend(item);

        while (DOM.sentimentList.children.length > 30) {
            DOM.sentimentList.removeChild(DOM.sentimentList.lastChild);
        }
    }

    // ===================== WEBSOCKET =====================
    function connectWS() {
        if (STATE.ws?.readyState === WebSocket.OPEN) return;
        if (STATE.reconnectTimer) return; // prevent duplicate reconnect attempts

        addMessage('جاري الاتصال بالخادم...', 'system');

        try {
            console.log('🔌 Connecting to WebSocket:', CONFIG.wsUrl);
            STATE.ws = new WebSocket(CONFIG.wsUrl);
        } catch (err) {
            console.error('❌ WebSocket connection error:', err);
            addMessage('فشل الاتصال بالخادم. تأكد من تشغيل البايثون backend على localhost:8765', 'system');
            updateConnectionStatus(false);
            // Reconnect after 5 seconds
            STATE.reconnectTimer = setTimeout(connectWS, 5000);
            return;
        }

        STATE.ws.binaryType = 'arraybuffer';

        STATE.ws.onopen = () => {
            console.log('✅ WebSocket connected');
            STATE.connected = true;
            updateConnectionStatus(true);
            clearTimeout(STATE.reconnectTimer);
            STATE.reconnectTimer = null;
            STATE.reconnectAttempts = 0;

            // Set initial dialect and gender
            sendJSON({ type: 'set_dialect', dialect: STATE.dialect });
            sendJSON({ type: 'set_gender', gender: STATE.gender });
        };

        STATE.ws.onclose = () => {
            console.log('⚠️ WebSocket closed');
            STATE.connected = false;
            updateConnectionStatus(false);
            addMessage('تم قطع الاتصال. إعادة الاتصال...', 'system');

            // Auto-reconnect with exponential backoff (3s, 5s, 8s, etc.)
            clearTimeout(STATE.reconnectTimer);
            const reconnectDelay = Math.min(30000, 3000 + (STATE.reconnectAttempts || 0) * 2000);
            STATE.reconnectAttempts = (STATE.reconnectAttempts || 0) + 1;
            STATE.reconnectTimer = setTimeout(connectWS, reconnectDelay);
        };

        STATE.ws.onerror = (event) => {
            console.error('❌ WebSocket error:', event);
            addMessage('خطأ في الاتصال بالخادم', 'system');
        };

        STATE.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                handleWSMessage(data);
            } catch (e) {
                console.error('Failed to parse WS message:', e);
            }
        };
    }

    function sendJSON(data) {
        if (STATE.ws?.readyState === WebSocket.OPEN) {
            STATE.ws.send(JSON.stringify(data));
            return true;
        }
        return false;
    }

    function sendAudioData(audioBytes) {
        if (STATE.ws?.readyState === WebSocket.OPEN) {
            STATE.ws.send(audioBytes);
        }
    }

    function handleWSMessage(data) {
        switch (data.type) {
            case 'connected':
                updateGenderUI(data.current_gender || STATE.gender);
                if (DOM.sessionInfo) {
                    const sid = data.session_id || '--';
                    DOM.sessionInfo.textContent = `Session: ${sid}`;
                }
                addMessage('تم الاتصال بنجاح! اختر اللهجة وابدأ المحادثة.', 'system');
                break;

            case 'dialect_changed':
                addMessage(`تم تغيير اللهجة إلى: ${DIALECT_NAMES[data.dialect]}`, 'system');
                break;

            case 'gender_changed':
                updateGenderUI(data.gender);
                addMessage(`تم تغيير الصوت إلى: ${data.gender === 'male' ? 'ذكر' : 'أنثى'}`, 'system');
                break;

            case 'bot_text':
                // Remove thinking indicator
                const thinking = document.querySelector('.thinking-msg');
                if (thinking) thinking.remove();
                STATE.lastBotMessageEl = addMessage(data.text, 'bot', { dialect: data.dialect });
                STATE.lastBotMessageEl.__ttsSegments = [];
                STATE.lastBotMessageEl.__ttsSegmentKeys = new Set();
                STATE.activeResponseId = data.response_id || null;
                STATE.lastBotNormalizedText = normalizeTranscript(data.text || '');
                break;

            case 'user_text':
                addSentimentEntry(data.text, data.analysis);
                break;

            case 'stt_result':
                if (data.text && data.text.trim()) {
                    addMessage(data.text.trim(), 'user');
                }
                break;

            case 'tts_audio':
                if (
                    data.response_id &&
                    STATE.activeResponseId &&
                    data.response_id !== STATE.activeResponseId
                ) {
                    break;
                }

                STATE.ttsServerDone = false;

                if (STATE.lastBotMessageEl) {
                    if (!Array.isArray(STATE.lastBotMessageEl.__ttsSegments)) {
                        STATE.lastBotMessageEl.__ttsSegments = [];
                    }

                    const segIndex = Number.isInteger(data.segment_index) ? data.segment_index : null;
                    const segKey = segIndex === null
                        ? `${STATE.lastBotMessageEl.__ttsSegments.length}-${(data.audio_base64 || '').length}`
                        : `${segIndex}-${data.response_id || 'noid'}`;

                    if (!STATE.lastBotMessageEl.__ttsSegmentKeys) {
                        STATE.lastBotMessageEl.__ttsSegmentKeys = new Set();
                    }
                    if (STATE.lastBotMessageEl.__ttsSegmentKeys.has(segKey)) {
                        break;
                    }
                    STATE.lastBotMessageEl.__ttsSegmentKeys.add(segKey);

                    STATE.lastBotMessageEl.__ttsSegments.push({
                        audio: data.audio_base64,
                        format: normalizeAudioFormat(data.format || 'mp3'),
                    });
                }

                enqueueTTSAudio(data.audio_base64, data.format || 'mp3');
                break;

            case 'tts_complete':
                if (
                    data.response_id &&
                    STATE.activeResponseId &&
                    data.response_id !== STATE.activeResponseId
                ) {
                    break;
                }
                STATE.ttsServerDone = true;
                finalizeTTSIfIdle();
                break;

            case 'bargein_detected':
                stopTTSPlayback();
                addMessage('تم إيقاف الصوت - تفضل تكلم', 'system');
                break;

            case 'interrupted':
                stopTTSPlayback();
                break;

            case 'vad_state':
                updateVADVisualizer(data.energy, data.is_speech);
                break;

            case 'speech_started':
                updateVADStatus('يتحدث...');
                DOM.vadStatusLabel?.classList.add('speaking');
                break;

            case 'speech_ended':
                updateVADStatus('صامت');
                DOM.vadStatusLabel?.classList.remove('speaking');
                break;

            case 'process_speech':
                // Backend sent audio for client-side STT
                processRecognizedSpeech(data.audio_base64);
                break;

            case 'error':
                addMessage(`خطأ: ${data.message}`, 'system');
                break;
        }
    }

    // ===================== TTS =====================
    async function replayBotMessage(messageEl, text) {
        const cachedSegments = Array.isArray(messageEl?.__ttsSegments)
            ? messageEl.__ttsSegments
            : [];

        if (cachedSegments.length > 0) {
            stopTTSPlayback();
            STATE.ttsServerDone = true;
            cachedSegments.forEach((segment) => {
                enqueueTTSAudio(segment.audio, segment.format || 'mp3');
            });
            return;
        }

        await speakText(text, { playbackOnly: true });
    }

    async function speakText(text, options = {}) {
        const playbackOnly = options.playbackOnly === true;

        // Only send chat text to the backend when this is a real user message.
        if (STATE.connected && !playbackOnly) {
            sendJSON({ type: 'text', content: text });
            return;
        }

        // Fallback: REST API
        try {
            const response = await fetch(`${CONFIG.apiUrl}/api/tts`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: text,
                    dialect: STATE.dialect,
                    gender: STATE.gender,
                }),
            });

            if (response.ok) {
                const data = await response.json();
                stopTTSPlayback();
                STATE.ttsServerDone = true;
                enqueueTTSAudio(data.audio_base64, data.audio_format || 'mp3');
            }
        } catch (e) {
            // Final fallback: browser TTS
            browserTTS(text);
        }
    }

    function enqueueTTSAudio(base64Audio, format = 'mp3') {
        if (!base64Audio) return;

        STATE.ttsQueue.push({
            audio: base64Audio,
            format: normalizeAudioFormat(format),
        });
        STATE.isTTSPlaying = true;
        showInterruptBtn();

        if (!STATE.currentAudio) {
            playNextTTSAudio();
        }
    }

    function playNextTTSAudio() {
        if (STATE.currentAudio) return;

        if (STATE.ttsQueue.length === 0) {
            finalizeTTSIfIdle();
            return;
        }

        const nextItem = STATE.ttsQueue.shift();
        const audio = new Audio(`data:audio/${nextItem.format};base64,${nextItem.audio}`);

        STATE.currentAudio = audio;
        STATE.isTTSPlaying = true;
        STATE.suppressRecognitionUntil = Date.now() + CONFIG.recognitionSuppressAfterTTSMs;

        audio.addEventListener('ended', () => {
            STATE.currentAudio = null;
            STATE.suppressRecognitionUntil = Date.now() + CONFIG.recognitionSuppressAfterTTSMs;
            playNextTTSAudio();
        });

        audio.addEventListener('error', () => {
            STATE.currentAudio = null;
            playNextTTSAudio();
        });

        audio.play().catch(e => {
            console.warn('Audio playback failed:', e);
            STATE.currentAudio = null;
            playNextTTSAudio();
        });
    }

    function finalizeTTSIfIdle() {
        if (STATE.currentAudio || STATE.ttsQueue.length > 0 || !STATE.ttsServerDone) {
            return;
        }

        STATE.isTTSPlaying = false;
        STATE.lastTTSFinishedAt = Date.now();
        STATE.suppressRecognitionUntil = STATE.lastTTSFinishedAt + CONFIG.recognitionSuppressAfterTTSMs;
        hideInterruptBtn();

        // Resume listening only when TTS stream is fully done.
        if (STATE.continuousMode && !STATE.isListening) {
            startListening();
        }
    }

    function stopTTSPlayback() {
        if (STATE.currentAudio) {
            STATE.currentAudio.pause();
            STATE.currentAudio.currentTime = 0;
            STATE.currentAudio = null;
        }
        STATE.ttsQueue = [];
        STATE.ttsServerDone = true;
        STATE.isTTSPlaying = false;
        STATE.lastTTSFinishedAt = Date.now();
        STATE.suppressRecognitionUntil = Date.now() + CONFIG.recognitionSuppressAfterTTSMs;
        hideInterruptBtn();
    }

    function browserTTS(text) {
        if (!('speechSynthesis' in window)) return;
        speechSynthesis.cancel();

        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = 'ar-SA';

        const voices = speechSynthesis.getVoices();
        const arVoice = voices.find(v => v.lang.startsWith('ar'));
        if (arVoice) utterance.voice = arVoice;

        speechSynthesis.speak(utterance);
    }

    // ===================== VAD + RECORDING =====================
    async function startListening() {
        if (STATE.isListening) return;

        try {
            STATE.stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: CONFIG.sampleRate,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                }
            });

            STATE.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: CONFIG.sampleRate,
            });

            STATE.audioInput = STATE.audioContext.createMediaStreamSource(STATE.stream);
            STATE.processor = STATE.audioContext.createScriptProcessor(4096, 1, 1);

            STATE.audioInput.connect(STATE.processor);
            STATE.processor.connect(STATE.audioContext.destination);

            // Reset VAD state
            STATE.isSpeaking = false;
            STATE.silenceFrames = 0;
            STATE.speechFrames = 0;
            STATE.audioBuffer = [];
            STATE.bufferLength = 0;

            STATE.processor.onaudioprocess = (e) => {
                if (!STATE.isListening) return;

                const inputData = e.inputBuffer.getChannelData(0);
                const pcm16 = float32ToPCM16(inputData);

                // Client-side VAD processing
                processVADLocally(inputData, pcm16);

                // Send raw audio to backend for server-side VAD
                if (STATE.connected) {
                    sendAudioData(pcm16.buffer);
                }
            };

            STATE.isListening = true;
            showVADVisualizer();
            updateMicButton('listening');
            DOM.micLabel.textContent = 'يستمع...';

        } catch (err) {
            console.error('Mic access failed:', err);
            addMessage('لا يمكن الوصول للميكروفون. تأكد من الأذونات.', 'system');
        }
    }

    function stopListening() {
        STATE.isListening = false;

        if (STATE.processor && STATE.audioInput) {
            STATE.audioInput.disconnect();
            STATE.processor.disconnect();
        }

        if (STATE.stream) {
            STATE.stream.getTracks().forEach(t => t.stop());
        }

        // If speech was in progress, finalize it
        if (STATE.isSpeaking && STATE.audioBuffer.length > 0) {
            finalizeSpeech();
        }

        STATE.isSpeaking = false;
        hideVADVisualizer();
        updateMicButton('idle');
        DOM.micLabel.textContent = 'ميكروفون';
    }

    function processVADLocally(float32Data, pcm16Data) {
        // Calculate RMS energy
        let sumSquares = 0;
        for (let i = 0; i < float32Data.length; i++) {
            sumSquares += float32Data[i] * float32Data[i];
        }
        const rms = Math.sqrt(sumSquares / float32Data.length);

        const isSpeech = rms > CONFIG.vadEnergyThreshold;
        const isBargeInSpeech = rms > CONFIG.bargeInThreshold;

        // Update visualizer
        updateVADVisualizer(rms, isSpeech);

        // Barge-in detection during TTS
        if (STATE.isTTSPlaying) {
            if (isBargeInSpeech) {
                STATE.speechFrames++;
                if (STATE.speechFrames >= CONFIG.bargeInFrames) {
                    stopTTSPlayback();
                    addMessage('تم المقاطعة - تفضل تكلم', 'system');
                    STATE.speechFrames = 0;

                    if (STATE.connected) {
                        sendJSON({ type: 'interrupt' });
                    }
                }
            } else {
                STATE.speechFrames = 0;
            }

            // Do not run normal VAD while bot speech is active.
            return;
        }

        // Normal VAD processing
        const framesForSilence = Math.ceil(CONFIG.vadSilenceMs / (4096 / CONFIG.sampleRate * 1000));
        const framesForSpeech = Math.ceil(CONFIG.vadSpeechMinMs / (4096 / CONFIG.sampleRate * 1000));

        if (isSpeech) {
            STATE.speechFrames++;
            STATE.silenceFrames = 0;

            if (!STATE.isSpeaking && STATE.speechFrames >= framesForSpeech) {
                STATE.isSpeaking = true;
                STATE.audioBuffer = [];
                STATE.bufferLength = 0;
                updateMicButton('recording');
                updateVADStatus('يتحدث...');
                DOM.vadStatusLabel?.classList.add('speaking');
            }
        } else {
            STATE.silenceFrames++;

            if (STATE.isSpeaking && STATE.silenceFrames >= framesForSilence) {
                // Speech ended
                STATE.isSpeaking = false;
                STATE.speechFrames = 0;
                finalizeSpeech();
                updateMicButton('listening');
                updateVADStatus('صامت');
                DOM.vadStatusLabel?.classList.remove('speaking');
            }
        }

        // Accumulate audio during speech
        if (STATE.isSpeaking) {
            const chunk = new Int16Array(pcm16Data);
            STATE.audioBuffer.push(chunk);
            STATE.bufferLength += chunk.length;
        }
    }

    function finalizeSpeech() {
        if (STATE.audioBuffer.length === 0) return;

        // Flatten audio buffer
        const result = new Int16Array(STATE.bufferLength);
        let offset = 0;
        for (const chunk of STATE.audioBuffer) {
            result.set(chunk, offset);
            offset += chunk.length;
        }

        STATE.audioBuffer = [];
        STATE.bufferLength = 0;

        // Convert to WAV blob for speech recognition
        const wavBlob = createWAVBlob(result, CONFIG.sampleRate);

        // Use browser Speech Recognition for STT
        recognizeSpeechFromBlob(wavBlob);
    }

    function recognizeSpeechFromBlob(wavBlob) {
        // SpeechRecognition onresult already sends final transcripts.
        // Keeping this as a no-op avoids duplicate submissions.
        void wavBlob;
    }

    function processRecognizedSpeech(audioBase64) {
        // Backend may emit process_speech events; browser STT is already active.
        void audioBase64;
    }

    // ===================== SPEECH RECOGNITION =====================
    function initSpeechRecognition() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) return false;

        STATE.recognition = new SpeechRecognition();
        STATE.recognition.lang = 'ar-EG';
        STATE.recognition.interimResults = true;
        STATE.recognition.continuous = true;
        STATE.recognition.maxAlternatives = 1;

        STATE._lastTranscript = '';

        STATE.recognition.onresult = (event) => {
            let finalTranscript = '';
            let interimTranscript = '';

            for (let i = event.resultIndex; i < event.results.length; i++) {
                const transcript = event.results[i][0].transcript;
                if (event.results[i].isFinal) {
                    finalTranscript += transcript;
                } else {
                    interimTranscript += transcript;
                }
            }

            if (finalTranscript) {
                const now = Date.now();
                const cleanedTranscript = finalTranscript.trim();
                const normalizedTranscript = normalizeTranscript(cleanedTranscript);

                if (!normalizedTranscript) {
                    return;
                }

                // Ignore recognizer echoes while TTS is playing (or right after it ends).
                if (STATE.isTTSPlaying || now < STATE.suppressRecognitionUntil) {
                    STATE._lastTranscript = '';
                    return;
                }

                // Prevent recognizer echoes from re-sending the assistant voice.
                if (isLikelyBotEcho(normalizedTranscript)) {
                    STATE._lastTranscript = '';
                    return;
                }

                // Ignore duplicate final hypotheses emitted back-to-back by some browsers.
                if (
                    normalizedTranscript === STATE.lastSentTranscript &&
                    now - STATE.lastSentAt < CONFIG.recognitionDedupMs
                ) {
                    return;
                }

                STATE.lastSentTranscript = normalizedTranscript;
                STATE.lastSentAt = now;
                STATE._lastTranscript = cleanedTranscript;

                handleUserMessage(cleanedTranscript);
            }

            // Show interim in input field
            if (interimTranscript) {
                DOM.textInput.value = interimTranscript;
            }
        };

        STATE.recognition.onend = () => {
            STATE.isRecognizing = false;
            // Restart if in continuous mode
            if (STATE.isListening && STATE.continuousMode) {
                try {
                    STATE.recognition.start();
                    STATE.isRecognizing = true;
                } catch (e) { /* already started */ }
            }
        };

        STATE.recognition.onerror = (e) => {
            if (e.error !== 'no-speech' && e.error !== 'aborted') {
                console.error('Speech recognition error:', e.error);
            }
        };

        return true;
    }

    function startSpeechRecognition() {
        if (!STATE.recognition) return;
        if (STATE.isRecognizing) return;

        try {
            STATE.recognition.start();
            STATE.isRecognizing = true;
        } catch (e) { /* already started */ }
    }

    function stopSpeechRecognition() {
        if (!STATE.recognition) return;
        try {
            STATE.recognition.stop();
            STATE.isRecognizing = false;
        } catch (e) { /* */ }
    }

    // ===================== MESSAGE HANDLING =====================
    function handleUserMessage(text) {
        if (!text.trim()) return;

        DOM.textInput.value = '';
        addMessage(text, 'user');
        const thinking = addThinking();

        if (STATE.connected) {
            // Send to backend
            sendJSON({ type: 'text', content: text });
        } else {
            // Offline response
            setTimeout(() => {
                thinking.remove();
                const reply = getOfflineReply(text);
                addMessage(reply, 'bot', { dialect: STATE.dialect });
                speakText(reply);
            }, 500);
        }
    }

    function getOfflineReply(text) {
        const t = text.trim();

        const greetings = ['السلام عليكم', 'أهلا', 'مرحبا', 'هاي', 'صباح الخير', 'مساء الخير'];
        if (greetings.some(g => t.includes(g))) {
            const responses = {
                cairene: 'وعليكم السلام! أهلاً وسهلاً، إزيك النهارده؟',
                saidi: 'وعليكم السلام يا أخي! عامل إيه؟',
                alexandrian: 'وعليكم السلام يا باشا! عامل إيه النهارده؟',
                bedouin: 'وعليكم السلام يا خوي! كيف الحال؟',
            };
            return responses[STATE.dialect] || responses.cairene;
        }

        if (t.includes('من انت') || t.includes('من أنت') || t.includes('مين انت')) {
            const responses = {
                cairene: 'أنا سيرفيا، المساعد الذكي بتاعك! عايز أساعدك في إيه النهارده؟',
                saidi: 'أنا سيرفيا يا أخي، المساعد الذكي. تحت أمرك!',
                alexandrian: 'أنا سيرفيا يا معلم! المساعد الذكي بتاعك. قول لي إيه اللي عايزه!',
                bedouin: 'أنا سيرفيا يا خوي، المساعد الذكي. أهلاً وسهلاً بك!',
            };
            return responses[STATE.dialect] || responses.cairene;
        }

        const defaults = {
            cairene: `تمام، فهمتك! قلت: "${t}". عايز حاجة تانية؟`,
            saidi: `فهمت عليك يا أخي! "${t}". عاوز حاجة تانية؟`,
            alexandrian: `حاضر يا باشا! فهمت: "${t}". عايز حاجة تانية؟`,
            bedouin: `فهمت عليك يا خوي! "${t}". تبغى شي ثاني؟`,
        };
        return defaults[STATE.dialect] || defaults.cairene;
    }

    // ===================== AUDIO UTILS =====================
    function float32ToPCM16(float32Array) {
        const pcm16 = new Int16Array(float32Array.length);
        for (let i = 0; i < float32Array.length; i++) {
            let s = Math.max(-1, Math.min(1, float32Array[i]));
            pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        return pcm16;
    }

    function createWAVBlob(pcm16Data, sampleRate) {
        const buffer = new ArrayBuffer(44 + pcm16Data.length * 2);
        const view = new DataView(buffer);

        const writeStr = (offset, str) => {
            for (let i = 0; i < str.length; i++) {
                view.setUint8(offset + i, str.charCodeAt(i));
            }
        };

        writeStr(0, 'RIFF');
        view.setUint32(4, 36 + pcm16Data.length * 2, true);
        writeStr(8, 'WAVE');
        writeStr(12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, 1, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        writeStr(36, 'data');
        view.setUint32(40, pcm16Data.length * 2, true);

        for (let i = 0; i < pcm16Data.length; i++) {
            view.setInt16(44 + i * 2, pcm16Data[i], true);
        }

        return new Blob([view], { type: 'audio/wav' });
    }

    // ===================== UI UPDATES =====================
    function updateConnectionStatus(connected) {
        STATE.connected = connected;
        if (DOM.statusDot) {
            DOM.statusDot.className = `status-dot ${connected ? 'connected' : 'disconnected'}`;
        }
        if (DOM.statusText) {
            DOM.statusText.textContent = connected ? 'متصل' : 'غير متصل';
        }
    }

    function updateMicButton(state) {
        if (!DOM.micBtn) return;
        DOM.micBtn.classList.remove('listening', 'recording');
        if (state === 'listening') DOM.micBtn.classList.add('listening');
        if (state === 'recording') DOM.micBtn.classList.add('recording');
    }

    function showVADVisualizer() {
        if (DOM.vadVisualizer) DOM.vadVisualizer.classList.add('active');
    }

    function hideVADVisualizer() {
        if (DOM.vadVisualizer) DOM.vadVisualizer.classList.remove('active');
    }

    function updateVADVisualizer(energy, isSpeech) {
        // Update energy meter
        if (DOM.vadEnergyFill) {
            const pct = Math.min(energy * 500, 100);
            DOM.vadEnergyFill.style.width = pct + '%';
            DOM.vadEnergyFill.classList.toggle('high', isSpeech);
        }

        // Update bars
        if (DOM.vadBars) {
            const bars = DOM.vadBars.children;
            const barCount = bars.length;
            for (let i = 0; i < barCount; i++) {
                const intensity = energy * (200 + Math.random() * 300);
                const h = isSpeech
                    ? Math.max(3, Math.min(40, intensity * (0.5 + Math.random())))
                    : Math.max(3, Math.min(8, 3 + Math.random() * 5));
                bars[i].style.height = h + 'px';
                bars[i].classList.toggle('speech', isSpeech);
            }
        }
    }

    function updateVADStatus(text) {
        if (DOM.vadStatusLabel) DOM.vadStatusLabel.textContent = text;
    }

    function showInterruptBtn() {
        if (DOM.interruptBtn) DOM.interruptBtn.classList.add('visible');
    }

    function hideInterruptBtn() {
        if (DOM.interruptBtn) DOM.interruptBtn.classList.remove('visible');
    }

    function updateDialectUI(dialect) {
        DOM.dialectBtns.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.dialect === dialect);
        });
    }

    function updateGenderUI(gender) {
        if (!DOM.genderToggle) return;

        const normalizedGender = gender === 'male' ? 'male' : 'female';
        STATE.gender = normalizedGender;

        const icon = DOM.genderToggle.querySelector('i');
        const text = DOM.genderToggle.querySelector('span');

        if (icon) {
            icon.className = normalizedGender === 'female' ? 'fas fa-venus' : 'fas fa-mars';
        }
        if (text) {
            text.textContent = normalizedGender === 'female' ? 'أنثى' : 'ذكر';
        }
    }

    // ===================== EVENT HANDLERS =====================
    function setupEvents() {
        // Dialect buttons
        DOM.dialectBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                STATE.dialect = btn.dataset.dialect;
                updateDialectUI(STATE.dialect);
                if (STATE.connected) {
                    sendJSON({ type: 'set_dialect', dialect: STATE.dialect });
                } else {
                    addMessage(`تم تغيير اللهجة إلى: ${DIALECT_NAMES[STATE.dialect]}`, 'system');
                }
            });
        });

        // Gender toggle
        if (DOM.genderToggle) {
            DOM.genderToggle.addEventListener('click', () => {
                const nextGender = STATE.gender === 'female' ? 'male' : 'female';
                updateGenderUI(nextGender);

                if (STATE.connected) {
                    sendJSON({ type: 'set_gender', gender: STATE.gender });
                }
            });
        }

        // Mic button
        if (DOM.micBtn) {
            DOM.micBtn.addEventListener('click', () => {
                if (STATE.isListening) {
                    stopListening();
                    stopSpeechRecognition();
                } else {
                    startListening();
                    startSpeechRecognition();
                }
            });
        }

        // Continuous mode toggle
        if (DOM.continuousToggle) {
            DOM.continuousToggle.addEventListener('click', () => {
                STATE.continuousMode = !STATE.continuousMode;
                DOM.continuousToggle.classList.toggle('active', STATE.continuousMode);

                if (STATE.continuousMode) {
                    addMessage('تم تفعيل الوضع المستمر - تحدث بحرية وسيتم الكشف عن كلامك تلقائياً', 'system');
                    startListening();
                    startSpeechRecognition();
                } else {
                    addMessage('تم إيقاف الوضع المستمر', 'system');
                    stopListening();
                    stopSpeechRecognition();
                }
            });
        }

        // Send button
        if (DOM.sendBtn) {
            DOM.sendBtn.addEventListener('click', () => {
                const text = DOM.textInput.value.trim();
                if (text) {
                    handleUserMessage(text);
                }
            });
        }

        // Enter to send
        if (DOM.textInput) {
            DOM.textInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    const text = DOM.textInput.value.trim();
                    if (text) {
                        handleUserMessage(text);
                    }
                }
            });

            DOM.textInput.addEventListener('input', function () {
                this.style.height = 'auto';
                this.style.height = Math.min(this.scrollHeight, 100) + 'px';
            });
        }

        // Interrupt button
        if (DOM.interruptBtn) {
            DOM.interruptBtn.addEventListener('click', () => {
                stopTTSPlayback();
                if (STATE.connected) {
                    sendJSON({ type: 'interrupt' });
                }
                addMessage('تم مقاطعة الصوت', 'system');
            });
        }

        // Clear
        if (DOM.clearBtn) {
            DOM.clearBtn.addEventListener('click', () => {
                DOM.messages.innerHTML = '';
                if (DOM.sentimentList) DOM.sentimentList.innerHTML = '';
                if (DOM.sentimentEmpty) DOM.sentimentEmpty.style.display = 'block';
                updateCurrentSentimentPill('neutral');
                addMessage('تم مسح المحادثة', 'system');
            });
        }
    }

    // ===================== INIT =====================
    function initVADBars() {
        if (!DOM.vadBars) return;
        DOM.vadBars.innerHTML = '';
        for (let i = 0; i < CONFIG.visualizerBars; i++) {
            const bar = document.createElement('div');
            bar.className = 'vad-bar';
            bar.style.height = '3px';
            DOM.vadBars.appendChild(bar);
        }
    }

    function init() {
        console.log('Servia Voice Chat initializing...');

        initVADBars();
        initSpeechRecognition();
        setupEvents();
        updateDialectUI(STATE.dialect);
        updateConnectionStatus(false);

        // Welcome
        addMessage('مرحباً بك في نظام الصوت المصري من سيرفيا!', 'bot', { dialect: STATE.dialect });
        addMessage('اختر اللهجة المصرية المناسبة وابدأ المحادثة الصوتية.', 'system');
        addMessage('يمكنك تفعيل "الوضع المستمر" للتحدث بحرية والمقاطعة في أي وقت.', 'system');

        // Auto-connect to backend
        connectWS();
    }

    // Start
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();
