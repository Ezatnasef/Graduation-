(function () {
    'use strict';

    // HARDCODED backend connection (always localhost:8765)
    // Do NOT depend on window.location since frontend might be served from http server on different port
    const BACKEND_HOST = 'localhost:8765';
    const BACKEND_HTTPS = false; // Set to true if backend is on HTTPS
    const API_BASE = `http://${BACKEND_HOST}`;
    const POLL_MS = 7000;

    const dom = {
        periodSelect: document.getElementById('periodSelect'),
        refreshBtn: document.getElementById('refreshBtn'),
        kpiMessages: document.getElementById('kpiMessages'),
        kpiActive: document.getElementById('kpiActive'),
        kpiEscRate: document.getElementById('kpiEscRate'),
        kpiLatency: document.getElementById('kpiLatency'),
        kpiConfidence: document.getElementById('kpiConfidence'),
        kpiSessions: document.getElementById('kpiSessions'),
        sentimentBars: document.getElementById('sentimentBars'),
        sentimentHint: document.getElementById('sentimentHint'),
        dialectBars: document.getElementById('dialectBars'),
        timelineChart: document.getElementById('timelineChart'),
        intentList: document.getElementById('intentList'),
        alertsList: document.getElementById('alertsList'),
        generatedAt: document.getElementById('generatedAt'),
        uptimeMeta: document.getElementById('uptimeMeta'),
    };

    let pollTimer = null;

    const arMap = {
        sentiments: {
            positive: 'إيجابي',
            angry: 'غاضب',
            sad: 'حزين',
            frustrated: 'متضايق',
            concerned: 'قلقان',
            neutral: 'محايد',
        },
        dialects: {
            cairene: 'قاهرية',
            saidi: 'صعيدية',
            alexandrian: 'إسكندرانية',
            bedouin: 'بدوية',
        },
        intents: {
            complaint: 'شكوى',
            inquiry: 'استفسار',
            request: 'طلب',
            cancellation_or_refund: 'إلغاء/استرجاع',
            technical_issue: 'مشكلة تقنية',
            praise: 'إشادة',
            greeting: 'ترحيب',
            feedback: 'ملاحظات',
            other: 'عام',
        },
        urgency: {
            low: 'منخفض',
            medium: 'متوسط',
            high: 'مرتفع',
        },
    };

    function safeText(value, fallback = '--') {
        if (value === null || value === undefined || value === '') return fallback;
        return String(value);
    }

    function makeBarRows(container, series, map = {}) {
        container.innerHTML = '';
        if (!Array.isArray(series) || series.length === 0) {
            container.innerHTML = '<div class="empty">لا توجد بيانات كافية للفترة المختارة.</div>';
            return;
        }

        series.forEach((item) => {
            const row = document.createElement('div');
            row.className = 'bar-row';

            const label = document.createElement('div');
            label.className = 'bar-label';
            label.textContent = map[item.label] || item.label;

            const track = document.createElement('div');
            track.className = 'bar-track';

            const fill = document.createElement('div');
            fill.className = 'bar-fill';
            fill.style.width = `${Math.max(2, Number(item.percent) || 0)}%`;
            track.appendChild(fill);

            const value = document.createElement('div');
            value.className = 'bar-value';
            value.textContent = `${item.count} (${item.percent}%)`;

            row.appendChild(label);
            row.appendChild(track);
            row.appendChild(value);
            container.appendChild(row);
        });
    }

    function drawTimeline(series) {
        dom.timelineChart.innerHTML = '';
        if (!Array.isArray(series) || series.length === 0) {
            dom.timelineChart.innerHTML = '<div class="empty">لا يوجد نشاط حتى الآن.</div>';
            return;
        }

        const last12 = series.slice(-12);
        const maxCount = Math.max(...last12.map((x) => Number(x.count) || 0), 1);

        last12.forEach((entry) => {
            const col = document.createElement('div');
            col.className = 'time-col';

            const bar = document.createElement('div');
            bar.className = 'time-bar';
            bar.style.height = `${Math.max(8, ((Number(entry.count) || 0) / maxCount) * 120)}px`;
            bar.title = `${entry.hour}: ${entry.count}`;

            const label = document.createElement('div');
            label.className = 'time-label';
            label.textContent = safeText(entry.hour, '--').slice(0, 5);

            col.appendChild(bar);
            col.appendChild(label);
            dom.timelineChart.appendChild(col);
        });
    }

    function renderIntentList(series) {
        dom.intentList.innerHTML = '';
        if (!Array.isArray(series) || series.length === 0) {
            dom.intentList.innerHTML = '<li class="empty">لا توجد نيات مرصودة بعد.</li>';
            return;
        }

        series.slice(0, 6).forEach((item) => {
            const li = document.createElement('li');
            li.textContent = `${arMap.intents[item.label] || item.label}: ${item.count} (${item.percent}%)`;
            dom.intentList.appendChild(li);
        });
    }

    function renderAlerts(alerts) {
        dom.alertsList.innerHTML = '';
        if (!Array.isArray(alerts) || alerts.length === 0) {
            dom.alertsList.innerHTML = '<li class="empty">لا توجد تنبيهات حالياً.</li>';
            return;
        }

        alerts.slice(0, 8).forEach((alert) => {
            const li = document.createElement('li');
            if (alert.urgency === 'high' || alert.needs_human_agent) {
                li.classList.add('high');
            }
            const urgency = arMap.urgency[alert.urgency] || alert.urgency;
            const sentiment = arMap.sentiments[alert.sentiment] || alert.sentiment;
            const intent = arMap.intents[alert.intent] || alert.intent;
            li.textContent = `${safeText(alert.time)} | ${safeText(alert.session_id)} | ${intent} | ${sentiment} | ${urgency}`;
            dom.alertsList.appendChild(li);
        });
    }

    function applyData(data) {
        const kpis = data.kpis || {};
        const dist = data.distributions || {};

        dom.kpiMessages.textContent = safeText(kpis.messages_window, '0');
        dom.kpiActive.textContent = safeText(kpis.active_sessions, '0');
        dom.kpiEscRate.textContent = `${safeText(kpis.escalation_rate_window, '0')}%`;
        dom.kpiLatency.textContent = `${safeText(kpis.avg_response_latency_ms, '0')} ms`;
        dom.kpiConfidence.textContent = safeText(kpis.avg_confidence_window, '0');
        dom.kpiSessions.textContent = safeText(kpis.total_sessions, '0');

        makeBarRows(dom.sentimentBars, dist.sentiment || [], arMap.sentiments);
        makeBarRows(dom.dialectBars, dist.dialect || [], arMap.dialects);
        drawTimeline(data.timeline || []);
        renderIntentList(dist.intent || []);
        renderAlerts(data.recent_alerts || []);

        const period = safeText(data.period_hours, '24');
        dom.sentimentHint.textContent = `آخر ${period} ساعة`;
        dom.generatedAt.textContent = `آخر تحديث: ${safeText(data.generated_at)}`;
        dom.uptimeMeta.textContent = `Uptime: ${safeText(data.service_uptime_minutes, '0')} دقيقة`;
    }

    async function fetchSummary() {
        const hours = Number(dom.periodSelect.value || 24);
        try {
            const res = await fetch(`${API_BASE}/api/analytics/summary?hours=${hours}`);
            if (!res.ok) {
                throw new Error(`HTTP ${res.status}`);
            }
            const data = await res.json();
            applyData(data);
        } catch (err) {
            dom.generatedAt.textContent = 'فشل تحميل التحليلات. تأكد إن الخدمة شغالة والـ endpoint صحيح.';
            console.error(err);
        }
    }

    function startPolling() {
        if (pollTimer) clearInterval(pollTimer);
        pollTimer = setInterval(fetchSummary, POLL_MS);
    }

    function init() {
        dom.refreshBtn.addEventListener('click', fetchSummary);
        dom.periodSelect.addEventListener('change', () => {
            fetchSummary();
            startPolling();
        });

        fetchSummary();
        startPolling();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
