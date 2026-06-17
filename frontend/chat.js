
(function() {
'use strict';

const CONFIG = {
wsUrl: 'wss://5000-dep-01kfhbtwr8tpjvfzh4348v1jre-d.cloudspaces.litng.ai/chat/query/talabat',
    reconnectDelay: 3000,
    sampleRate: 16000,
    debugMode: true
};



const STATE = {
    mode: 'offline', // 'offline' or 'online'
    ttsEnabled: true,
    isRecording: false,
    isListening: false,
    
    // WebSocket
    ws: null,
    wsReconnectTimeout: null,
    
    // Speech Recognition (Offline)
    recognition: null,
    
    // Audio Recording (Online & Offline when using WebSocket)
    stream: null,
    audioContext: null,
    processor: null,
    audioInput: null,
    leftChannelData: [],
    recordingLength: 0
};


const $ = (id) => document.getElementById(id);
const DOM = {
    messages: $('messagesContainer'),
    input: $('messageInput'),
    sendBtn: $('sendBtn'),
    micBtn: $('micBtn'),
    modeToggle: $('modeToggle'),
    ttsToggle: $('ttsToggle'),
    clearBtn: $('clearBtn'),
    statusDot: $('statusDot'),
    statusText: $('statusText')
};

function formatTime() {
    return new Date().toLocaleTimeString('ar-EG', {
        hour: '2-digit',
        minute: '2-digit'
    });
}

function scrollToBottom() {
    DOM.messages.scrollTop = DOM.messages.scrollHeight;
}

function detectLanguage(text) {
    const hasArabic = /[\u0600-\u06FF]/.test(text);
    const hasEnglish = /[a-zA-Z]/.test(text);
    return {
        hasArabic,
        hasEnglish,
        isPrimaryArabic: hasArabic && !hasEnglish,
        isPrimaryEnglish: hasEnglish && !hasArabic
    };
}

function addMessage(text, type = 'bot', meta = '') {
    const msg = document.createElement('div');
    msg.className = `message ${type}`;
    
    if (meta) {
        const metaDiv = document.createElement('div');
        metaDiv.className = 'message-meta';
        metaDiv.innerHTML = `<strong>${meta}</strong>`;
        msg.appendChild(metaDiv);
    }
    
    const content = document.createElement('div');
    content.textContent = text;
    msg.appendChild(content);
    
    const timestamp = document.createElement('div');
    timestamp.className = 'timestamp';
    timestamp.textContent = formatTime();
    msg.appendChild(timestamp);
    
    // Click to speak for bot messages
    if (type === 'bot') {
        msg.title = 'اضغط للاستماع';
        msg.addEventListener('click', () => speak(text));
    }
    
    DOM.messages.appendChild(msg);
    scrollToBottom();
    
    return msg;
}

function addAudioMessage(base64Audio, isUser = false) {
    const msg = document.createElement('div');
    msg.className = `message ${isUser ? 'user' : 'bot'}`;
    
    const audio = document.createElement('audio');
    audio.className = 'audio-player';
    audio.controls = true;
    audio.src = `data:audio/wav;base64,${base64Audio}`;
    msg.appendChild(audio);
    
    const timestamp = document.createElement('div');
    timestamp.className = 'timestamp';
    timestamp.textContent = formatTime();
    msg.appendChild(timestamp);
    
    DOM.messages.appendChild(msg);
    scrollToBottom();
    
    // Auto-play
    audio.play().catch(e => console.warn('تعذر تشغيل الصوت تلقائياً:', e));
}

function updateStatus() {
    const modeIcon = DOM.modeToggle.querySelector('i');
    const modeText = DOM.modeToggle.querySelector('span');
    
    if (STATE.mode === 'offline') {
        DOM.statusDot.className = 'status-dot offline';
        DOM.statusText.textContent = 'محلي';
        modeIcon.className = 'fas fa-desktop';
        modeText.textContent = 'محلي';
        DOM.modeToggle.classList.remove('active');
        DOM.modeToggle.classList.add('offline-mode');
    } else {
        const wsState = STATE.ws?.readyState;
        const isConnected = wsState === WebSocket.OPEN;
        const isConnecting = wsState === WebSocket.CONNECTING;
        
        DOM.statusDot.className = `status-dot ${isConnected ? 'online' : 'offline'}`;
        
        if (isConnected) {
            DOM.statusText.textContent = 'متصل ✓';
        } else if (isConnecting) {
            DOM.statusText.textContent = 'جاري الاتصال...';
        } else {
            DOM.statusText.textContent = 'غير متصل ✗';
        }
        
        modeIcon.className = 'fas fa-globe';
        modeText.textContent = 'متصل';
        DOM.modeToggle.classList.add('active');
        DOM.modeToggle.classList.remove('offline-mode');
        
        if (CONFIG.debugMode) {
            const states = ['CONNECTING', 'OPEN', 'CLOSING', 'CLOSED'];
            console.log(`WebSocket State: ${wsState !== undefined ? states[wsState] : 'NULL'}`);
        }
    }
}


function generateBotReply(userText) {
    const thinking = addMessage('...', 'bot');
    thinking.classList.add('thinking');
    
    setTimeout(() => {
        thinking.remove();
        const reply = getSmartReply(userText);
        addMessage(reply, 'bot');
        if (STATE.ttsEnabled) speak(reply);
    }, 400 + Math.random() * 600);
}

function getSmartReply(text) {
    const t = text.trim();
    const lower = t.toLowerCase();
    const lang = detectLanguage(t);
    
    // Arabic greetings
    const arabicGreetings = ['السلام عليكم', 'أهلا', 'مرحبا', 'هاي', 'صباح الخير', 'مساء الخير'];
    if (arabicGreetings.some(g => t.includes(g))) {
        return 'وعليكم السلام! أهلاً بك، كيف يمكنني مساعدتك اليوم؟ 😊';
    }
    
    // English greetings
    if (/^(hi|hello|hey|good morning|good evening|howdy|what's up|wassup)$/i.test(lower)) {
        return 'Hello! How can I assist you today? 👋';
    }
    
    // Who are you - English
    if (lower.includes('who are you') || lower.includes('what are you') || lower.includes('who r u')) {
        return 'I am a smart assistant that works in two modes: Offline (local processing in your browser) and Online (connected to AI server via WebSocket). You can switch between them easily! 🤖';
    }
    
    // Who are you - Arabic
    if (t.includes('من انت') || t.includes('من أنت') || t.includes('مين انت')) {
        return 'أنا مساعد ذكي يعمل بوضعين: محلي (Offline) باستخدام متصفحك، ومتصل (Online) عبر خادم AI. يمكنك التبديل بينهما بسهولة! 🤖';
    }
    
    // Company - English
    if (lower.includes('company') || lower.includes('made you') || lower.includes('created you') || lower.includes('developed you')) {
        return 'I was developed by Servia 🚀';
    }
    
    // Company - Arabic
    if (t.includes('الشركة') || t.includes('صنعتك') || t.includes('طورتك')) {
        return 'تم تطويري بواسطة سيرفيا (Servia) 🚀';
    }
    
    // Help - English
    if (lower.includes('help') || lower === 'help me') {
        return 'I can help you in several ways:\n• Answering questions\n• Text and voice chat\n• Switching between offline and online modes\nAsk me anything! 💡';
    }
    
    // Help - Arabic
    if (t.includes('مساعدة') || t.includes('ساعدني')) {
        return 'يمكنني مساعدتك بعدة طرق:\n• الإجابة على الأسئلة\n• المحادثة النصية والصوتية\n• التبديل بين الوضع المحلي والمتصل\nجرب أن تسألني أي شيء! 💡';
    }
    
    // Name questions - English
    if (lower.includes('your name') || lower.includes('what do i call you')) {
        return 'You can call me Assistant! I\'m here to help you with voice and text chat. 😊';
    }
    
    // Name questions - Arabic
    if (t.includes('اسمك') || t.includes('ماذا اسمك')) {
        return 'يمكنك أن تناديني المساعد! أنا هنا لمساعدتك في المحادثة الصوتية والنصية. 😊';
    }
    
    // Questions - respond in same language
    if (t.endsWith('؟') || t.endsWith('?')) {
        if (lang.isPrimaryEnglish) {
            const questions = [
                `That's a great question! Here's a brief answer: "${t}"`,
                `Let me think... Possible answer: "${t}"`,
                `That depends on context, but generally speaking: "${t}"`
            ];
            return questions[Math.floor(Math.random() * questions.length)];
        } else {
            const questions = [
                `سؤال رائع! إليك إجابة مختصرة: "${t}"`,
                `دعني أفكر... الإجابة المحتملة: "${t}"`,
                `هذا يعتمد على السياق، لكن بشكل عام: "${t}"`
            ];
            return questions[Math.floor(Math.random() * questions.length)];
        }
    }
    
    // Default - respond in same language
    if (lang.isPrimaryEnglish) {
        const responses = [
            `Got it: "${t}" - How can I help you further?`,
            `Thanks for your message! You said: "${t}"`,
            `Message received successfully: "${t}"`
        ];
        return responses[Math.floor(Math.random() * responses.length)];
    } else {
        const responses = [
            `فهمت: "${t}" - كيف يمكنني مساعدتك أكثر؟`,
            `شكراً على رسالتك! قلت: "${t}"`,
            `رسالتك وصلت بنجاح: "${t}"`
        ];
        return responses[Math.floor(Math.random() * responses.length)];
    }
}


function initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        console.warn('التعرف على الصوت غير مدعوم');
        return false;
    }
    
    try {
        STATE.recognition = new SpeechRecognition();
        STATE.recognition.lang = 'ar-SA';
        STATE.recognition.interimResults = true;
        STATE.recognition.continuous = false;
        
        STATE.recognition.onresult = (event) => {
            let transcript = '';
            for (let i = 0; i < event.results.length; i++) {
                transcript += event.results[i][0].transcript;
            }
            DOM.input.value = transcript;
        };
        
        STATE.recognition.onend = () => {
            STATE.isListening = false;
            DOM.micBtn.classList.remove('recording');
        };
        
        STATE.recognition.onerror = (e) => {
            console.error('خطأ في التعرف على الصوت:', e);
            STATE.isListening = false;
            DOM.micBtn.classList.remove('recording');
        };
        
        return true;
    } catch (e) {
        console.error('فشل تهيئة التعرف على الصوت:', e);
        return false;
    }
}

function toggleSpeechRecognition() {
    if (!STATE.recognition) {
        addMessage('⚠️ التعرف على الصوت غير مدعوم في متصفحك', 'system');
        return;
    }
    
    if (STATE.isListening) {
        try {
            STATE.recognition.stop();
        } catch (e) {
            console.warn('خطأ في إيقاف التعرف:', e);
        }
        STATE.isListening = false;
        DOM.micBtn.classList.remove('recording');
    } else {
        try {
            STATE.recognition.start();
            STATE.isListening = true;
            DOM.micBtn.classList.add('recording');
        } catch (e) {
            console.error('فشل بدء التعرف:', e);
            addMessage('⚠️ فشل بدء التعرف على الصوت', 'system');
        }
    }
}

async function startRecording() {
    try {
        STATE.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        STATE.audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: CONFIG.sampleRate
        });
        
        STATE.audioInput = STATE.audioContext.createMediaStreamSource(STATE.stream);
        STATE.processor = STATE.audioContext.createScriptProcessor(4096, 1, 1);
        
        STATE.audioInput.connect(STATE.processor);
        STATE.processor.connect(STATE.audioContext.destination);
        
        STATE.leftChannelData = [];
        STATE.recordingLength = 0;
        
        STATE.processor.onaudioprocess = (e) => {
            if (!STATE.isRecording) return;
            
            const inputData = e.inputBuffer.getChannelData(0);
            const chunk = new Float32Array(inputData);
            STATE.leftChannelData.push(chunk);
            STATE.recordingLength += chunk.length;
        };
        
        STATE.isRecording = true;
        DOM.micBtn.classList.add('recording');
        
    } catch (err) {
        console.error('فشل الوصول للميكروفون:', err);
        addMessage('⚠️ لا يمكن الوصول للميكروفون. تأكد من الأذونات.', 'system');
    }
}

function stopRecording() {
    STATE.isRecording = false;
    DOM.micBtn.classList.remove('recording');
    
    if (STATE.processor && STATE.audioInput) {
        STATE.audioInput.disconnect();
        STATE.processor.disconnect();
    }
    
    if (STATE.stream) {
        STATE.stream.getTracks().forEach(track => track.stop());
    }
    
    // Convert to WAV
    const pcmData = flattenArray(STATE.leftChannelData, STATE.recordingLength);
    const wavBlob = encodeWAV(pcmData, STATE.audioContext.sampleRate);
    
    // Convert to base64 and send
    const reader = new FileReader();
    reader.readAsDataURL(wavBlob);
    reader.onloadend = () => {
        const base64 = reader.result.split(',')[1];
        
        if (sendToWebSocket('audio', base64)) {
            addMessage('🎤 تم إرسال رسالة صوتية', 'user');
        }
    };
}

function flattenArray(channelBuffer, recordingLength) {
    const result = new Float32Array(recordingLength);
    let offset = 0;
    for (let i = 0; i < channelBuffer.length; i++) {
        const buffer = channelBuffer[i];
        result.set(buffer, offset);
        offset += buffer.length;
    }
    return result;
}

function encodeWAV(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);
    
    const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };
    
    // RIFF identifier
    writeString(0, 'RIFF');
    // file length
    view.setUint32(4, 36 + samples.length * 2, true);
    // RIFF type
    writeString(8, 'WAVE');
    // format chunk identifier
    writeString(12, 'fmt ');
    // format chunk length
    view.setUint32(16, 16, true);
    // sample format (raw)
    view.setUint16(20, 1, true);
    // channel count
    view.setUint16(22, 1, true);
    // sample rate
    view.setUint32(24, sampleRate, true);
    // byte rate (sample rate * block align)
    view.setUint32(28, sampleRate * 2, true);
    // block align (channel count * bytes per sample)
    view.setUint16(32, 2, true);
    // bits per sample
    view.setUint16(34, 16, true);
    // data chunk identifier
    writeString(36, 'data');
    // data chunk length
    view.setUint32(40, samples.length * 2, true);
    
    // PCM samples
    floatTo16BitPCM(view, 44, samples);
    
    return new Blob([view], { type: 'audio/wav' });
}

function floatTo16BitPCM(output, offset, input) {
    for (let i = 0; i < input.length; i++, offset += 2) {
        let s = Math.max(-1, Math.min(1, input[i]));
        s = s < 0 ? s * 0x8000 : s * 0x7FFF;
        output.setInt16(offset, s, true);
    }
}

function connectWebSocket() {
    if (STATE.ws?.readyState === WebSocket.OPEN) {
        console.log('⚠️ WebSocket already connected');
        return;
    }
    
    console.log(`🔌 Connecting to WebSocket: ${CONFIG.wsUrl}`);
    addMessage(`🔌 جاري الاتصال بالخادم...`, 'system');
    
    try {
        STATE.ws = new WebSocket(CONFIG.wsUrl);
    } catch (err) {
        console.error('❌ Failed to create WebSocket:', err);
        addMessage(`❌ خطأ في الاتصال: ${err.message}`, 'system');
        addMessage('💡 تأكد من تشغيل الـ backend على المنفذ الصحيح', 'system');
        updateStatus();
        return;
    }
    
    STATE.ws.onopen = () => {
        console.log('✅ WebSocket متصل بنجاح!');
        addMessage('✅ تم الاتصال بالخادم بنجاح', 'system');
        updateStatus();
        clearTimeout(STATE.wsReconnectTimeout);
    };
    
    STATE.ws.onclose = (event) => {
        console.log('❌ WebSocket مقطوع - Code:', event.code, 'Reason:', event.reason);
        addMessage('❌ تم قطع الاتصال بالخادم', 'system');
        updateStatus();
        
        if (STATE.mode === 'online') {
            console.log(`🔄 سيتم إعادة الاتصال بعد ${CONFIG.reconnectDelay}ms`);
            STATE.wsReconnectTimeout = setTimeout(connectWebSocket, CONFIG.reconnectDelay);
        }
    };
    
    STATE.ws.onerror = (err) => {
        console.error('❌ WebSocket Error:', err);
        addMessage('❌ خطأ في الاتصال بالخادم', 'system');
        addMessage('💡 تأكد من:', 'system');
        addMessage(`   • Backend شغال على: ${CONFIG.wsUrl}`, 'system');
        addMessage('   • لا يوجد Firewall يمنع الاتصال', 'system');
    };
    
    STATE.ws.onmessage = (event) => {
        try {
            if (CONFIG.debugMode) {
                console.log('📥 Raw message:', event.data);
            }
            const data = event.data;
            handleWebSocketMessage(data);
        } catch (e) {
            console.error('❌ فشل تحليل رسالة WebSocket:', e);
            addMessage(`⚠️ رسالة غير صالحة من الخادم`, 'system');
        }
    };
}

function handleWebSocketMessage(data) {
    console.log('📥 Received from backend:', data);
    addMessage(data, 'bot');
    // Backend response format: { type: "response_text"|"response_audio"|"text"|"error", content: "..." }
    // const { type, content } = data;
    
    // switch (type) {
    //     case 'response_text':
    //         addMessage(content, 'bot');
    //         if (STATE.ttsEnabled) speak(content);
    //         break;
            
    //     case 'text':
    //         // Echo from server (like "You said: ...")
    //         if (!content?.startsWith('You said:')) {
    //             addMessage(content, 'bot');
    //             if (STATE.ttsEnabled) speak(content);
    //         }
    //         break;
            
    //     case 'response_audio':
    //         addAudioMessage(content, false);
    //         break;
            
    //     case 'error':
    //         addMessage(`⚠️ خطأ: ${content}`, 'system');
    //         break;
            
    //     default:
    //         console.warn('نوع رسالة غير معروف:', type);
    // }
}

function sendToWebSocket(type, content) {
    if (!STATE.ws) {
        console.error('❌ WebSocket غير موجود');
        addMessage('⚠️ لا يوجد اتصال بالخادم. التبديل للوضع المحلي...', 'system');
        switchToOfflineMode();
        return false;
    }
    
    if (STATE.ws.readyState !== WebSocket.OPEN) {
        console.error('❌ WebSocket not connected. ReadyState:', STATE.ws.readyState);
        const states = ['CONNECTING', 'OPEN', 'CLOSING', 'CLOSED'];
        addMessage(`⚠️ الاتصال في حالة: ${states[STATE.ws.readyState]}`, 'system');
        
        if (STATE.ws.readyState === WebSocket.CONNECTING) {
            addMessage('⏳ انتظر حتى يكتمل الاتصال...', 'system');
        } else {
            addMessage('⚠️ التبديل للوضع المحلي...', 'system');
            switchToOfflineMode();
        }
        return false;
    }

    const message = {
    "type":type,
    "content":content
};

    if (CONFIG.debugMode) {
        console.log('📤 Sending to backend:', message);
    }
    
    try {
        STATE.ws.send(JSON.stringify(message));
        console.log('✅ Message sent successfully');
        return true;
    } catch (err) {
        console.error('❌ Failed to send message:', err);
        addMessage(`❌ فشل إرسال الرسالة: ${err.message}`, 'system');
        return false;
    }
}


function speak(text) {
    if (!STATE.ttsEnabled || !('speechSynthesis' in window)) return;
    
    speechSynthesis.cancel();
    
    const utterance = new SpeechSynthesisUtterance(text);
    
    // Detect language
    utterance.lang = /[\u0600-\u06FF]/.test(text) ? 'ar-SA' : 'en-US';
    
    const setVoice = () => {
        const voices = speechSynthesis.getVoices();
        if (voices.length) {
            const targetLang = utterance.lang.split('-')[0];
            utterance.voice = voices.find(v => v.lang.startsWith(targetLang)) || voices[0];
        }
        
        try {
            speechSynthesis.speak(utterance);
        } catch (e) {
            console.warn('فشل النطق:', e);
        }
    };
    
    const voices = speechSynthesis.getVoices();
    if (voices.length) {
        setVoice();
    } else {
        speechSynthesis.addEventListener('voiceschanged', setVoice, { once: true });
    }
}


function switchToOnlineMode() {
    STATE.mode = 'online';
    connectWebSocket();
    updateStatus();
    addMessage('🌐 تم التبديل إلى الوضع المتصل (Online)', 'system');
}

function switchToOfflineMode() {
    STATE.mode = 'offline';
    
    if (STATE.ws) {
        STATE.ws.close();
        STATE.ws = null;
    }
    
    clearTimeout(STATE.wsReconnectTimeout);
    updateStatus();
    addMessage('💻 تم التبديل إلى الوضع المحلي (Offline)', 'system');
}

function toggleMode() {
    if (STATE.mode === 'offline') {
        switchToOnlineMode();
    } else {
        switchToOfflineMode();
    }
}

// ============================================
// MESSAGE HANDLING
// ============================================
function sendMessage() {
    const text = DOM.input.value.trim();
    if (!text) return;
    
    addMessage(text, 'user');
    DOM.input.value = '';
    DOM.input.style.height = 'auto';
    
    if (STATE.mode === 'online') {
        if (!sendToWebSocket('text', text)) {
            // Fallback to offline if connection failed
            generateBotReply(text);
        }
    } else {
        generateBotReply(text);
    }
}

function handleMicClick() {
    if (STATE.mode === 'offline') {
        // Use Speech Recognition in offline mode
        toggleSpeechRecognition();
    } else {
        // Use WAV recording in online mode
        if (STATE.isRecording) {
            stopRecording();
        } else {
            startRecording();
        }
    }
}


DOM.sendBtn.addEventListener('click', sendMessage);

DOM.input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

DOM.input.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 120) + 'px';
});

DOM.micBtn.addEventListener('click', handleMicClick);

DOM.modeToggle.addEventListener('click', toggleMode);

DOM.ttsToggle.addEventListener('click', () => {
    STATE.ttsEnabled = !STATE.ttsEnabled;
    const icon = DOM.ttsToggle.querySelector('i');
    const text = DOM.ttsToggle.querySelector('span');
    
    if (STATE.ttsEnabled) {
        icon.className = 'fas fa-volume-up';
        text.textContent = 'نطق';
        DOM.ttsToggle.classList.add('active');
    } else {
        icon.className = 'fas fa-volume-mute';
        text.textContent = 'صامت';
        DOM.ttsToggle.classList.remove('active');
    }
});

DOM.clearBtn.addEventListener('click', () => {
    DOM.messages.innerHTML = '';
    addMessage('✨ تم مسح المحادثة', 'system');
});


function init() {
    console.log('🚀 بدء تهيئة التطبيق...');
    
    // Initialize speech recognition
    const recognitionAvailable = initSpeechRecognition();
    
    // Set initial status
    updateStatus();
    
    // Welcome messages
    addMessage('👋 مرحباً بك في الشات الموحد!', 'bot');
    addMessage('هذا التطبيق يعمل بوضعين:', 'bot');
    addMessage('💻 محلي: معالجة فورية في المتصفح (Speech Recognition + TTS)', 'bot');
    addMessage('🌐 متصل: اتصال بخادم AI عبر WebSocket', 'bot');
    addMessage('اضغط على زر "💻 محلي" للتبديل بين الأوضاع', 'system');
    
    if (!recognitionAvailable) {
        addMessage('⚠️ ملاحظة: التعرف على الصوت غير مدعوم في متصفحك الحالي', 'system');
    }
    
    console.log('✅ التطبيق جاهز!');
}

init();

})();