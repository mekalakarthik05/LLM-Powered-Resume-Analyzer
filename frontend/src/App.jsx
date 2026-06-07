import React, { useState, useRef, useEffect } from 'react';

/* ─────────────────────────────────────────────
   Helpers
───────────────────────────────────────────── */
const asArray = (v) => (Array.isArray(v) ? v : []);
const clamp = (v) => Math.max(0, Math.min(Number(v) || 0, 100));
const fmt = (v) => (typeof v === 'number' && v <= 1 ? `${Math.round(v * 100)}%` : v || '—');
const termText = (v) => (typeof v === 'string' ? v : v?.term || v?.name || '');
const evidenceText = (v) => (typeof v === 'string' ? v : v?.snippet || v?.text || '');
const titleCase = (v) =>
  String(v || '').split(' ').filter(Boolean)
    .map((p) => p[0].toUpperCase() + p.slice(1)).join(' ');

const scoreColor = (s) => s >= 80 ? '#0F6E56' : s >= 50 ? '#854F0B' : '#791F1F';
const scoreBg = (s) => s >= 80 ? '#E1F5EE' : s >= 50 ? '#FAEEDA' : '#FCEBEB';
const scoreBorder = (s) => s >= 80 ? '#5DCAA5' : s >= 50 ? '#EF9F27' : '#F09595';
const barColor = (s) => s >= 70 ? '#1D9E75' : s >= 45 ? '#BA7517' : '#A32D2D';
const scoreLabel = (s) => s >= 80 ? 'Excellent match' : s >= 60 ? 'Strong candidate' : s >= 40 ? 'Moderate fit' : 'Significant gaps';
const DEFAULT_REQUEST_TIMEOUT_MS = 30 * 60 * 1000;
const configuredRequestTimeoutMs = Number(import.meta.env.VITE_REQUEST_TIMEOUT);
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const REQUEST_TIMEOUT_MS =
  Number.isFinite(configuredRequestTimeoutMs) && configuredRequestTimeoutMs > 0
    ? configuredRequestTimeoutMs
    : DEFAULT_REQUEST_TIMEOUT_MS;
const filenameFromDisposition = (header) => {
  const match = /filename\*?=(?:UTF-8''|")?([^";\n]+)/i.exec(header || '');
  return match ? decodeURIComponent(match[1].replace(/"/g, '')) : '';
};

/* ─────────────────────────────────────────────
   Global styles (injected once)
───────────────────────────────────────────── */
const GLOBAL_CSS = `
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300;1,9..40,400&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { color-scheme: light dark; }
body {
  font-family: 'DM Sans', system-ui, sans-serif;
  background: #F7F6F2;
  color: #1A1A18;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  font-size: 14px;
  line-height: 1.5;
  letter-spacing: -0.01em;
}
@media (prefers-color-scheme: dark) {
  body { background: #141412; color: #F0EFE9; }
}

@keyframes aura-spin { to { transform: rotate(360deg); } }
@keyframes aura-fadeUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
@keyframes aura-pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.45; } }
@keyframes aura-bar { from { width: 0; } to { width: var(--w); } }
@keyframes aura-shimmer {
  0% { background-position: -600px 0; }
  100% { background-position: 600px 0; }
}

.aura-fadeUp { animation: aura-fadeUp 0.4s cubic-bezier(0.16,1,0.3,1) both; }
.aura-spin { animation: aura-spin 0.75s linear infinite; }

textarea, input { font-family: inherit; }
textarea:focus { outline: none; box-shadow: 0 0 0 3px rgba(24,95,165,0.12); }
button { font-family: inherit; cursor: pointer; }
button:disabled { cursor: not-allowed; }
input[type="file"] { display: none; }

textarea::-webkit-scrollbar,
pre::-webkit-scrollbar { width: 5px; }
textarea::-webkit-scrollbar-thumb,
pre::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.15); border-radius: 99px; }
textarea::-webkit-scrollbar-track,
pre::-webkit-scrollbar-track { background: transparent; }
`;

function useGlobalStyles() {
  useEffect(() => {
    const id = 'aura-global-styles';
    if (document.getElementById(id)) return;
    const tag = document.createElement('style');
    tag.id = id;
    tag.textContent = GLOBAL_CSS;
    document.head.appendChild(tag);
    return () => { const el = document.getElementById(id); if (el) el.remove(); };
  }, []);
}

/* ─────────────────────────────────────────────
   Design tokens
───────────────────────────────────────────── */
const T = {
  bg0: 'var(--aura-bg0, #FFFFFF)',
  bg1: 'var(--aura-bg1, #F7F6F2)',
  bg2: 'var(--aura-bg2, #EEEEE8)',
  text1: 'var(--aura-t1, #1A1A18)',
  text2: 'var(--aura-t2, #5C5C58)',
  text3: 'var(--aura-t3, #9C9C96)',
  border: 'rgba(0,0,0,0.07)',
  border2: 'rgba(0,0,0,0.12)',
  blue: '#185FA5',
  blueLt: '#EBF3FC',
  blueMd: '#7BB3E8',
  green: '#1D9E75',
  greenLt: '#E4F6F0',
  greenMd: '#5DCAA5',
  red: '#A32D2D',
  redLt: '#FCEAEA',
  redMd: '#EF9595',
  amber: '#BA7517',
  amberLt: '#FAF0DC',
  amberMd: '#EFA827',
  purple: '#534AB7',
  purpleLt: '#EEEDFE',
  purpleMd: '#AFA9EC',
};

/* ─────────────────────────────────────────────
   Primitive components
───────────────────────────────────────────── */
function Spinner({ size = 18, color }) {
  return (
    <div className="aura-spin" style={{
      width: size, height: size, borderRadius: '50%', flexShrink: 0,
      border: `${Math.max(1.5, size / 9)}px solid ${color ? color + '44' : 'rgba(0,0,0,0.12)'}`,
      borderTopColor: color || '#1A1A18',
    }} />
  );
}

function Card({ children, style, className = '' }) {
  return (
    <div className={`aura-fadeUp ${className}`} style={{
      background: T.bg0,
      border: `0.5px solid ${T.border2}`,
      borderRadius: 16,
      padding: '22px 24px',
      ...style,
    }}>
      {children}
    </div>
  );
}

function SectionLabel({ icon, title }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 7, marginBottom: 16 }}>
      <span style={{ display: 'flex', opacity: 0.35, flexShrink: 0 }}>{icon}</span>
      <span style={{ fontSize: 10, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.1em', color: T.text2 }}>{title}</span>
    </div>
  );
}

function Pill({ label, variant = 'blue' }) {
  const map = {
    blue: { bg: T.blueLt, color: '#0C447C', border: T.blueMd },
    green: { bg: T.greenLt, color: '#085041', border: T.greenMd },
    red: { bg: T.redLt, color: '#791F1F', border: T.redMd },
    amber: { bg: T.amberLt, color: '#633806', border: T.amberMd },
    purple: { bg: T.purpleLt, color: '#3C3489', border: T.purpleMd },
    gray: { bg: T.bg2, color: T.text2, border: T.border2 },
  };
  const s = map[variant] || map.blue;
  return (
    <span style={{
      display: 'inline-block', whiteSpace: 'nowrap',
      fontSize: 11, fontWeight: 500,
      padding: '3px 10px', borderRadius: 99,
      background: s.bg, color: s.color,
      border: `0.5px solid ${s.border}`,
    }}>{label}</span>
  );
}

function MetricMini({ label, value, sub }) {
  return (
    <div style={{ background: T.bg1, borderRadius: 10, padding: '11px 14px' }}>
      <div style={{ fontSize: 11, color: T.text3, marginBottom: 3 }}>{label}</div>
      <div style={{ fontSize: 19, fontWeight: 600, letterSpacing: '-0.03em', color: T.text1 }}>{value}</div>
      {sub && <div style={{ fontSize: 10, color: T.text3, marginTop: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{sub}</div>}
    </div>
  );
}

function ProgressBar({ value, color }) {
  return (
    <div style={{ height: 3, background: T.border2, borderRadius: 99, overflow: 'hidden', marginTop: 7 }}>
      <div style={{
        height: '100%', width: `${clamp(value)}%`,
        background: color, borderRadius: 99,
        transition: 'width 1s cubic-bezier(.4,0,.2,1)',
      }} />
    </div>
  );
}

/* ─────────────────────────────────────────────
   Score ring
───────────────────────────────────────────── */
function ScoreRing({ score }) {
  const r = 46;
  const circ = 2 * Math.PI * r;
  const offset = circ - (clamp(score) / 100) * circ;
  const col = scoreColor(score);
  return (
    <svg width="120" height="120" viewBox="0 0 120 120" style={{ display: 'block' }}>
      <circle cx="60" cy="60" r={r} fill="none" stroke={T.border2} strokeWidth="7" />
      <circle
        cx="60" cy="60" r={r} fill="none"
        stroke={col} strokeWidth="7"
        strokeDasharray={circ} strokeDashoffset={offset}
        strokeLinecap="round"
        transform="rotate(-90 60 60)"
        style={{ transition: 'stroke-dashoffset 1.2s cubic-bezier(.4,0,.2,1)' }}
      />
      <text x="60" y="55" textAnchor="middle" dominantBaseline="middle"
        style={{ fontSize: 28, fontWeight: 700, fill: col, fontFamily: "'DM Sans', system-ui, sans-serif", letterSpacing: '-0.04em' }}>
        {Math.round(score)}
      </text>
      <text x="60" y="74" textAnchor="middle"
        style={{ fontSize: 9, fill: T.text3, fontFamily: "'DM Sans', system-ui, sans-serif", letterSpacing: '0.12em' }}>
        SCORE
      </text>
    </svg>
  );
}

/* ─────────────────────────────────────────────
   Loading state
───────────────────────────────────────────── */
const DEFAULT_PROGRESS_STEPS = [
  { key: 'validate_input', label: 'Validate request' },
  { key: 'save_upload', label: 'Read uploaded resume' },
  { key: 'extract_text', label: 'Extract resume text' },
  { key: 'enrich_resume', label: 'Structure resume data' },
  { key: 'score_resume', label: 'Score ATS match' },
  { key: 'rag_chunk', label: 'Chunk resume for RAG' },
  { key: 'rag_jd_embedding', label: 'Embed job description' },
  { key: 'rag_resume_embeddings', label: 'Embed resume chunks' },
  { key: 'rag_rank', label: 'Rank relevant evidence' },
  { key: 'llm_insight', label: 'Run recruiter LLM' },
  { key: 'llm_parse', label: 'Parse AI insight' },
  { key: 'draft_resume', label: 'Generate optimized draft' },
  { key: 'package_report', label: 'Package final report' },
];

const buildProgressSteps = (steps = DEFAULT_PROGRESS_STEPS) =>
  steps.map((step, index) => ({
    key: step.key || step.label || String(index),
    label: step.label || step.key || `Step ${index + 1}`,
    status: 'pending',
    detail: '',
    meta: {},
  }));

const progressIsComplete = (status) => ['done', 'warning', 'error'].includes(status);

function mergeProgressEvent(prev, event) {
  if (event?.type === 'steps' && Array.isArray(event.steps)) {
    return buildProgressSteps(event.steps);
  }
  if (event?.type !== 'progress' || !event.key) return prev;

  const existing = prev.some((step) => step.key === event.key)
    ? prev
    : [...prev, { key: event.key, label: event.label || event.key, status: 'pending', detail: '', meta: {} }];

  const activeIndex = existing.findIndex((step) => step.key === event.key);
  return existing.map((step, index) => {
    if (step.key === event.key) {
      return {
        ...step,
        label: event.label || step.label,
        status: event.status || step.status,
        detail: event.detail || step.detail,
        meta: event.meta || step.meta,
        duration_ms: event.duration_ms ?? step.duration_ms,
      };
    }
    if (event.status === 'active' && index < activeIndex && ['pending', 'active'].includes(step.status)) {
      return { ...step, status: 'done' };
    }
    return step;
  });
}

const activeProgressStep = (steps) =>
  steps.find((step) => step.status === 'active')
  || [...steps].reverse().find((step) => progressIsComplete(step.status))
  || steps[0];

const progressPercent = (steps) => {
  if (!steps.length) return 0;
  const activeIndex = steps.findIndex((step) => step.status === 'active');
  const completed = steps.filter((step) => progressIsComplete(step.status)).length;
  const value = activeIndex >= 0 ? completed + 0.5 : completed;
  return Math.min(100, Math.round((value / steps.length) * 100));
};

async function readProgressStream(response, onEvent) {
  if (!response.body?.getReader) return null;

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let finalData = null;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      const event = JSON.parse(trimmed);
      if (event.type === 'error') {
        throw new Error(event.detail || 'Analysis failed.');
      }
      if (event.type === 'result') {
        finalData = event.data;
      } else {
        onEvent(event);
      }
    }
  }

  const tail = buffer.trim();
  if (tail) {
    const event = JSON.parse(tail);
    if (event.type === 'error') {
      throw new Error(event.detail || 'Analysis failed.');
    }
    if (event.type === 'result') {
      finalData = event.data;
    } else {
      onEvent(event);
    }
  }

  return finalData;
}

function LoadingState({ progressSteps }) {
  const active = activeProgressStep(progressSteps);
  const pct = progressPercent(progressSteps);

  return (
    <Card style={{ padding: '42px 32px', textAlign: 'center' }}>
      <Spinner size={34} color={T.blue} />
      <div style={{ marginTop: 22, fontSize: 17, fontWeight: 600, letterSpacing: '-0.02em' }}>
        {active?.label || 'Processing...'}
      </div>
      <div style={{ fontSize: 13, color: T.text2, marginTop: 4, marginBottom: 32 }}>
        {active?.detail || 'Waiting for the backend to report progress.'}
      </div>
      <div style={{ maxWidth: 620, margin: '0 auto', display: 'grid', gap: 8 }}>
        {progressSteps.map((s) => {
          const done = s.status === 'done';
          const activeStep = s.status === 'active';
          const warning = s.status === 'warning';
          const error = s.status === 'error';
          const rowBg = activeStep ? T.blueLt : warning ? T.amberLt : error ? T.redLt : done ? T.greenLt : T.bg1;
          const rowBorder = activeStep ? T.blueMd : warning ? T.amberMd : error ? T.redMd : done ? T.greenMd : T.border;
          const rowColor = activeStep ? '#0C447C' : warning ? '#633806' : error ? '#791F1F' : done ? '#085041' : T.text2;
          const iconBg = activeStep ? T.blue : warning ? T.amber : error ? T.red : done ? T.green : T.border2;
          return (
            <div key={s.key} style={{
              display: 'flex', alignItems: 'center', gap: 10,
              padding: '10px 14px', borderRadius: 8, textAlign: 'left', fontSize: 13,
              background: rowBg,
              border: `0.5px solid ${rowBorder}`,
              color: rowColor,
            }}>
              <div style={{
                width: 20, height: 20, borderRadius: '50%', flexShrink: 0,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                background: iconBg,
              }}>
                {(done || warning) && (
                  <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
                    <path d="M2 5l2 2 4-4" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                )}
                {error && <span style={{ color: '#fff', fontSize: 12, fontWeight: 700 }}>!</span>}
                {activeStep && <Spinner size={10} color="white" />}
              </div>
              <div style={{ minWidth: 0, flex: 1 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10, alignItems: 'baseline' }}>
                  <span style={{ fontWeight: activeStep ? 600 : 500 }}>{s.label}</span>
                  {s.duration_ms != null && (
                    <span style={{ fontSize: 10, color: rowColor, opacity: 0.7, whiteSpace: 'nowrap' }}>
                      {(s.duration_ms / 1000).toFixed(1)}s
                    </span>
                  )}
                </div>
                {s.detail && (
                  <div style={{ fontSize: 11, color: rowColor, opacity: 0.82, marginTop: 2, lineHeight: 1.45 }}>
                    {s.detail}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
      <div style={{
        height: 3, background: T.border, borderRadius: 99,
        maxWidth: 620, margin: '22px auto 0', overflow: 'hidden',
      }}>
        <div style={{
          height: '100%', width: `${pct}%`,
          background: T.blue, borderRadius: 99, transition: 'width 0.5s ease',
        }} />
      </div>
    </Card>
  );
}

/* ─────────────────────────────────────────────
   Empty state
───────────────────────────────────────────── */
function EmptyState() {
  return (
    <Card style={{ padding: '60px 32px', textAlign: 'center' }}>
      <div style={{
        width: 58, height: 58, borderRadius: 14,
        background: T.blueLt, border: `0.5px solid ${T.blueMd}`,
        display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 20px',
      }}>
        <svg width="26" height="26" viewBox="0 0 28 28" fill="none">
          <circle cx="14" cy="14" r="9.5" stroke={T.blue} strokeWidth="1.8" />
          <circle cx="14" cy="14" r="3.5" fill={T.blue} />
          <path d="M14 4v2.5M14 21.5V24M4 14h2.5M21.5 14H24" stroke={T.blue} strokeWidth="1.8" strokeLinecap="round" />
        </svg>
      </div>
      <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase', color: T.blue, marginBottom: 8 }}>
        AI engine ready
      </div>
      <div style={{ fontSize: 22, fontWeight: 700, letterSpacing: '-0.03em', marginBottom: 10 }}>
        Advanced resume intelligence
      </div>
      <div style={{ fontSize: 14, color: T.text2, lineHeight: 1.75, maxWidth: 400, margin: '0 auto 38px' }}>
        Paste a job description and upload your resume on the left. Our AI will analyse the match, surface gaps, and generate an optimised draft.
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 12, maxWidth: 480, margin: '0 auto' }}>
        {[
          {
            label: 'Industry analysis', sub: 'Role-specific benchmarks', bg: T.blueLt, icon: (
              <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
                <rect x="1" y="9" width="3" height="6" rx="1" fill={T.blue} opacity=".6" />
                <rect x="6" y="5" width="3" height="10" rx="1" fill={T.blue} opacity=".85" />
                <rect x="11" y="2" width="3" height="13" rx="1" fill={T.blue} />
              </svg>
            )
          },
          {
            label: 'ATS signals', sub: 'Keyword matching', bg: T.greenLt, icon: (
              <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
                <path d="M2 8V3a1 1 0 011-1h5l5 5-5 6-6-5z" stroke={T.green} strokeWidth="1.2" strokeLinejoin="round" />
                <circle cx="5.5" cy="5.5" r="1" fill={T.green} />
              </svg>
            )
          },
          {
            label: 'Smart optimisation', sub: 'Fact-preserving edits', bg: T.purpleLt, icon: (
              <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
                <path d="M8 1l1.5 5H15l-4.5 3 1.5 5L8 11l-4 3 1.5-5L1 6h5.5L8 1z" stroke={T.purple} strokeWidth="1.2" strokeLinejoin="round" />
              </svg>
            )
          },
        ].map(({ label, sub, bg, icon }) => (
          <div key={label} style={{ background: bg, borderRadius: 12, padding: '14px 12px' }}>
            <div style={{ width: 30, height: 30, borderRadius: 8, background: 'rgba(255,255,255,0.6)', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 8px' }}>{icon}</div>
            <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 2 }}>{label}</div>
            <div style={{ fontSize: 11, color: T.text2 }}>{sub}</div>
          </div>
        ))}
      </div>
    </Card>
  );
}

/* ─────────────────────────────────────────────
   Result sections
───────────────────────────────────────────── */
function ScoreHero({ result }) {
  const score = result.score || 0;
  const ind = result.industry_analysis;
  return (
    <Card>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr auto', gap: 24, alignItems: 'center' }}>
        <div>
          <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase', color: T.blue, marginBottom: 4 }}>
            Match analysis
          </div>
          <div style={{ fontSize: 21, fontWeight: 700, letterSpacing: '-0.03em', marginBottom: 6 }}>
            Resume intelligence report
          </div>
          <div style={{ fontSize: 13, color: T.text2, lineHeight: 1.7, marginBottom: 14, maxWidth: 480 }}>
            {result.scoring_model?.note || 'AI analysis complete. Review the findings below.'}
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {ind && <Pill label={ind.detected_industry} variant="blue" />}
            <Pill label={`v${result.scoring_model?.analysis_version || '4.0'}`} variant="purple" />
            <span style={{
              display: 'inline-block', whiteSpace: 'nowrap',
              fontSize: 11, fontWeight: 500, padding: '3px 10px', borderRadius: 99,
              background: scoreBg(score), color: scoreColor(score), border: `0.5px solid ${scoreBorder(score)}`,
            }}>{scoreLabel(score)}</span>
          </div>
        </div>
        <ScoreRing score={score} />
      </div>
    </Card>
  );
}

function ScoreBreakdown({ breakdown }) {
  if (!breakdown.length) return null;
  return (
    <Card>
      <SectionLabel title="Score breakdown" icon={
        <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
          <rect x="1" y="9" width="3" height="6" rx="1" fill="currentColor" opacity=".55" />
          <rect x="6" y="5" width="3" height="10" rx="1" fill="currentColor" opacity=".8" />
          <rect x="11" y="2" width="3" height="13" rx="1" fill="currentColor" />
        </svg>
      } />
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 10 }}>
        {breakdown.map((c) => {
          const s = clamp(c.score);
          const col = barColor(s);
          return (
            <div key={c.key || c.name} style={{ background: T.bg1, borderRadius: 10, padding: '13px 14px' }}>
              <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 2, letterSpacing: '-0.01em' }}>{c.name}</div>
              <div style={{ fontSize: 11, color: T.text2, lineHeight: 1.5, marginBottom: 8, minHeight: 28 }}>{c.why}</div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                <span style={{ fontSize: 20, fontWeight: 700, letterSpacing: '-0.04em', color: col }}>{s}</span>
                <span style={{ fontSize: 10, color: T.text3 }}>wt {fmt(c.weight)}</span>
              </div>
              <ProgressBar value={s} color={col} />
              {asArray(c.signals).slice(0, 2).map((sig, i) => (
                <div key={i} style={{ display: 'flex', gap: 5, marginTop: 7, fontSize: 11, color: T.text2, lineHeight: 1.4 }}>
                  <svg width="11" height="11" viewBox="0 0 12 12" fill="none" style={{ marginTop: 1, flexShrink: 0 }}>
                    <path d="M2 6l3 3 5-5" stroke={T.green} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                  {sig}
                </div>
              ))}
            </div>
          );
        })}
      </div>
    </Card>
  );
}

function TwoCol({ left, right }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 14 }}>
      {left}
      {right}
    </div>
  );
}

function AISummary({ result, warnings }) {
  const warningRows = asArray(warnings).filter(Boolean);
  return (
    <Card>
      <SectionLabel title="AI summary" icon={
        <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
          <path d="M8 2C5.24 2 3 4.24 3 7c0 1.5.66 2.85 1.7 3.78L5 14h6l.3-3.22A4.98 4.98 0 0013 7c0-2.76-2.24-5-5-5z" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round" />
          <path d="M6 9h4M7 11h2" stroke="currentColor" strokeWidth="1" strokeLinecap="round" />
        </svg>
      } />
      {warningRows.length > 0 && (
        <div style={{
          background: T.amberLt,
          border: `0.5px solid ${T.amberMd}`,
          borderRadius: 10,
          padding: '10px 12px',
          marginBottom: 12,
          display: 'grid',
          gap: 6,
        }}>
          {warningRows.map((warning, index) => (
            <div key={index} style={{ fontSize: 12, color: '#633806', lineHeight: 1.55 }}>
              {warning}
            </div>
          ))}
        </div>
      )}
      <p style={{ fontSize: 13, lineHeight: 1.75, color: T.text2, margin: 0 }}>
        {result.ai_insight?.summary || 'Analysis complete.'}
      </p>
    </Card>
  );
}

function MatchedKeywords({ matched }) {
  return (
    <Card>
      <SectionLabel title="Matched keywords" icon={
        <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
          <path d="M2 8V3a1 1 0 011-1h5l5 5-5 6-6-5z" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round" />
          <circle cx="5.5" cy="5.5" r="1" fill="currentColor" />
        </svg>
      } />
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
        {matched.length
          ? matched.slice(0, 20).map((t, i) => <Pill key={i} label={titleCase(termText(t))} variant="green" />)
          : <span style={{ fontSize: 13, color: T.text3 }}>None found</span>
        }
      </div>
    </Card>
  );
}

function MissingKeywords({ missing }) {
  return (
    <Card>
      <SectionLabel title="Missing keywords" icon={
        <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
          <path d="M8 2L1 14h14L8 2z" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round" />
          <path d="M8 7v3M8 12h.01" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
        </svg>
      } />
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
        {missing.length
          ? missing.slice(0, 22).map((t, i) => <Pill key={i} label={titleCase(termText(t))} variant="red" />)
          : <span style={{ fontSize: 13, color: T.text3 }}>No critical gaps</span>
        }
      </div>
    </Card>
  );
}

function ResumeEvidence({ evidence }) {
  return (
    <Card>
      <SectionLabel title="Resume evidence" icon={
        <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
          <path d="M4 2h6l3 3v9a1 1 0 01-1 1H4a1 1 0 01-1-1V3a1 1 0 011-1z" stroke="currentColor" strokeWidth="1.2" />
          <path d="M10 2v4h3M5 8h5M5 11h3" stroke="currentColor" strokeWidth="1" strokeLinecap="round" />
        </svg>
      } />
      <div style={{ display: 'grid', gap: 10 }}>
        {evidence.length
          ? evidence.slice(0, 6).map((item, i) => (
            <div key={i} style={{ borderLeft: `2px solid ${T.blueMd}`, paddingLeft: 11 }}>
              <div style={{ fontSize: 12, lineHeight: 1.65, color: T.text2 }}>{evidenceText(item)}</div>
              {termText(item) && (
                <div style={{ fontSize: 10, fontWeight: 600, color: T.blue, marginTop: 2, letterSpacing: '0.04em', textTransform: 'uppercase' }}>
                  {titleCase(termText(item))}
                </div>
              )}
            </div>
          ))
          : <span style={{ fontSize: 13, color: T.text3 }}>No snippets found</span>
        }
      </div>
    </Card>
  );
}

function StrategicRecs({ ag }) {
  if (!ag) return null;
  const sections = [
    { label: 'Immediate actions', items: ag.immediate_actions, bg: T.redLt, border: T.redMd, text: '#791F1F', icon: '!' },
    { label: 'Critical gaps', items: ag.critical_gaps, bg: T.amberLt, border: T.amberMd, text: '#633806', icon: '△' },
    { label: 'Long-term strategy', items: ag.long_term_strategy, bg: T.greenLt, border: T.greenMd, text: '#085041', icon: '↑' },
  ];
  return (
    <Card>
      <SectionLabel title="Strategic recommendations" icon={
        <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
          <circle cx="8" cy="7" r="3.5" stroke="currentColor" strokeWidth="1.2" />
          <path d="M8 1v1.5M8 13v1.5M1 7h1.5M13.5 7H15M6 10.5V13h4v-2.5" stroke="currentColor" strokeWidth="1.1" strokeLinecap="round" />
        </svg>
      } />
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12 }}>
        {sections.map(({ label, items, bg, border, text, icon }) => (
          <div key={label} style={{ background: bg, border: `0.5px solid ${border}`, borderRadius: 10, padding: '14px 15px' }}>
            <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', color: text, marginBottom: 10 }}>{label}</div>
            {asArray(items).slice(0, 3).map((item, i) => (
              <div key={i} style={{ display: 'flex', gap: 7, marginBottom: 8, fontSize: 12, color: text, lineHeight: 1.5 }}>
                <span style={{ flexShrink: 0, fontWeight: 600 }}>{icon}</span>
                {item}
              </div>
            ))}
          </div>
        ))}
      </div>
    </Card>
  );
}

function DetailedReview({ llm }) {
  if (!llm.length) return null;
  return (
    <Card>
      <SectionLabel title="Detailed review" icon={
        <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
          <path d="M8 1l1.5 5H15l-4.5 3 1.5 5L8 11l-4 3 1.5-5L1 6h5.5L8 1z" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round" />
        </svg>
      } />
      <div style={{ display: 'grid', gap: 10 }}>
        {llm.slice(0, 6).map((item, i) => (
          <div key={i} style={{
            background: T.bg1, borderRadius: 10, padding: '13px 15px',
            borderLeft: `2.5px solid ${T.purpleMd}`,
          }}>
            <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 3, letterSpacing: '-0.01em' }}>{item.signal}</div>
            <div style={{ fontSize: 12, color: T.text2, lineHeight: 1.65, marginBottom: 3 }}>{item.finding}</div>
            <div style={{ fontSize: 11, color: T.text3, lineHeight: 1.5, marginBottom: item.fix ? 6 : 0 }}>{item.why_it_matters}</div>
            {item.fix && (
              <div style={{ fontSize: 12, fontWeight: 500, color: T.green, display: 'flex', gap: 5 }}>
                <svg width="11" height="11" viewBox="0 0 12 12" fill="none" style={{ marginTop: 1, flexShrink: 0 }}>
                  <path d="M2 6l3 3 5-5" stroke={T.green} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                {item.fix}
              </div>
            )}
          </div>
        ))}
      </div>
    </Card>
  );
}

function latestTraceRows(trace) {
  const byKey = new Map();
  asArray(trace).forEach((event) => {
    if (event?.key) byKey.set(event.key, event);
  });

  const orderedKeys = new Set(DEFAULT_PROGRESS_STEPS.map((step) => step.key));
  const orderedRows = DEFAULT_PROGRESS_STEPS
    .map((step) => byKey.get(step.key))
    .filter(Boolean);
  const extraRows = [...byKey.entries()]
    .filter(([key]) => !orderedKeys.has(key))
    .map(([, event]) => event);

  return [...orderedRows, ...extraRows];
}

function ProcessingTrace({ trace, warnings }) {
  const rows = latestTraceRows(trace);
  const warningRows = asArray(warnings).filter(Boolean);
  if (!rows.length && !warningRows.length) return null;

  return (
    <Card>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10, alignItems: 'flex-start', flexWrap: 'wrap', marginBottom: 14 }}>
        <SectionLabel title="Processing trace" icon={
          <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
            <path d="M4 3h8M4 8h8M4 13h8" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
            <circle cx="2" cy="3" r="1" fill="currentColor" />
            <circle cx="2" cy="8" r="1" fill="currentColor" />
            <circle cx="2" cy="13" r="1" fill="currentColor" />
          </svg>
        } />
        {warningRows.length > 0 && <Pill label={`${warningRows.length} warning${warningRows.length === 1 ? '' : 's'}`} variant="amber" />}
      </div>

      {warningRows.length > 0 && (
        <div style={{ display: 'grid', gap: 8, marginBottom: 12 }}>
          {warningRows.map((warning, index) => (
            <div key={index} style={{
              background: T.amberLt,
              border: `0.5px solid ${T.amberMd}`,
              borderRadius: 8,
              padding: '9px 11px',
              fontSize: 12,
              color: '#633806',
              lineHeight: 1.55,
            }}>
              {warning}
            </div>
          ))}
        </div>
      )}

      <div style={{ display: 'grid', gap: 7 }}>
        {rows.map((row) => {
          const warning = row.status === 'warning';
          const error = row.status === 'error';
          const done = row.status === 'done';
          const color = error ? T.red : warning ? T.amber : done ? T.green : T.blue;
          const bg = error ? T.redLt : warning ? T.amberLt : done ? T.greenLt : T.blueLt;
          const border = error ? T.redMd : warning ? T.amberMd : done ? T.greenMd : T.blueMd;
          return (
            <div key={row.key} style={{
              display: 'grid',
              gridTemplateColumns: '10px minmax(0, 1fr) auto',
              gap: 10,
              alignItems: 'baseline',
              background: bg,
              border: `0.5px solid ${border}`,
              borderRadius: 8,
              padding: '9px 11px',
              color,
            }}>
              <span style={{ width: 7, height: 7, borderRadius: '50%', background: color, display: 'inline-block' }} />
              <div style={{ minWidth: 0 }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: T.text1 }}>{row.label}</div>
                {row.detail && <div style={{ fontSize: 11, color: T.text2, lineHeight: 1.45, marginTop: 1 }}>{row.detail}</div>}
              </div>
              <span style={{ fontSize: 10, color: T.text3, whiteSpace: 'nowrap' }}>
                {row.duration_ms != null ? `${(row.duration_ms / 1000).toFixed(1)}s` : row.status}
              </span>
            </div>
          );
        })}
      </div>
    </Card>
  );
}

const splitResumeLines = (text) =>
  String(text || '')
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

const normalizeDiffLine = (line) => line.replace(/\s+/g, ' ').trim().toLowerCase();

function buildResumeLineDiff(originalText, draftText) {
  const original = splitResumeLines(originalText);
  const draft = splitResumeLines(draftText);
  const originalNorm = original.map(normalizeDiffLine);
  const draftNorm = draft.map(normalizeDiffLine);
  const dp = Array.from({ length: original.length + 1 }, () => Array(draft.length + 1).fill(0));

  for (let i = original.length - 1; i >= 0; i -= 1) {
    for (let j = draft.length - 1; j >= 0; j -= 1) {
      dp[i][j] = originalNorm[i] === draftNorm[j]
        ? dp[i + 1][j + 1] + 1
        : Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }

  const ops = [];
  let i = 0;
  let j = 0;
  while (i < original.length && j < draft.length) {
    if (originalNorm[i] === draftNorm[j]) {
      ops.push({ type: 'equal', left: original[i], right: draft[j] });
      i += 1;
      j += 1;
    } else if (dp[i + 1][j] >= dp[i][j + 1]) {
      ops.push({ type: 'removed', left: original[i], right: '' });
      i += 1;
    } else {
      ops.push({ type: 'added', left: '', right: draft[j] });
      j += 1;
    }
  }
  while (i < original.length) {
    ops.push({ type: 'removed', left: original[i], right: '' });
    i += 1;
  }
  while (j < draft.length) {
    ops.push({ type: 'added', left: '', right: draft[j] });
    j += 1;
  }

  const rows = [];
  for (let k = 0; k < ops.length;) {
    if (ops[k].type === 'equal') {
      rows.push({ kind: 'equal', left: ops[k].left, right: ops[k].right });
      k += 1;
      continue;
    }

    const removed = [];
    const added = [];
    while (k < ops.length && ops[k].type !== 'equal') {
      if (ops[k].type === 'removed') removed.push(ops[k].left);
      if (ops[k].type === 'added') added.push(ops[k].right);
      k += 1;
    }

    const blockRows = Math.max(removed.length, added.length);
    for (let n = 0; n < blockRows; n += 1) {
      rows.push({
        kind: removed[n] && added[n] ? 'changed' : removed[n] ? 'removed' : 'added',
        left: removed[n] || '',
        right: added[n] || '',
      });
    }
  }

  return rows;
}

function DiffLineCell({ text, kind, side }) {
  const isLeft = side === 'left';
  const marked =
    (kind === 'removed' && isLeft) ||
    (kind === 'added' && !isLeft) ||
    kind === 'changed';
  const bg = !marked
    ? T.bg1
    : kind === 'removed'
      ? T.redLt
      : kind === 'added'
        ? T.greenLt
        : isLeft ? T.amberLt : T.greenLt;
  const border = !marked
    ? T.border
    : kind === 'removed'
      ? T.redMd
      : kind === 'added'
        ? T.greenMd
        : isLeft ? T.amberMd : T.greenMd;
  const marker = kind === 'equal' ? ' ' : kind === 'removed' ? '-' : kind === 'added' ? '+' : '~';

  return (
    <div style={{
      minHeight: 34,
      background: bg,
      border: `0.5px solid ${border}`,
      borderRadius: 8,
      padding: '8px 10px',
      display: 'grid',
      gridTemplateColumns: '16px minmax(0, 1fr)',
      gap: 7,
      alignItems: 'start',
    }}>
      <span style={{ fontFamily: "'DM Mono', monospace", fontSize: 11, color: marked ? T.text1 : T.text3 }}>
        {text ? marker : ' '}
      </span>
      <span style={{
        fontFamily: "'DM Mono', monospace",
        fontSize: 11,
        lineHeight: 1.55,
        color: text ? T.text1 : T.text3,
        whiteSpace: 'pre-wrap',
        wordBreak: 'break-word',
      }}>
        {text || ' '}
      </span>
    </div>
  );
}

function ResumeDiff({ original, draft }) {
  if (!original || !draft) return null;
  const rows = buildResumeLineDiff(original, draft);
  if (!rows.length) return null;

  const added = rows.filter((row) => row.kind === 'added').length;
  const removed = rows.filter((row) => row.kind === 'removed').length;
  const changed = rows.filter((row) => row.kind === 'changed').length;

  return (
    <Card>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10, flexWrap: 'wrap', alignItems: 'flex-start', marginBottom: 14 }}>
        <SectionLabel title="Resume diff" icon={
          <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
            <path d="M2 4h12M2 8h12M2 12h12" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
            <path d="M5 2v12M11 2v12" stroke="currentColor" strokeWidth="1" strokeLinecap="round" opacity=".55" />
          </svg>
        } />
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          <Pill label={`${changed} updated`} variant="amber" />
          <Pill label={`${added} added`} variant="green" />
          <Pill label={`${removed} removed`} variant="red" />
        </div>
      </div>

      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
        gap: 10,
        marginBottom: 8,
      }}>
        <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: T.text2 }}>
          Original resume
        </div>
        <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: T.text2 }}>
          Optimized draft
        </div>
      </div>

      <div style={{ maxHeight: 560, overflowY: 'auto', display: 'grid', gap: 8, paddingRight: 3 }}>
        {rows.map((row, index) => (
          <div key={`${row.kind}-${index}`} style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
            gap: 10,
          }}>
            <DiffLineCell text={row.left} kind={row.kind} side="left" />
            <DiffLineCell text={row.right} kind={row.kind} side="right" />
          </div>
        ))}
      </div>
    </Card>
  );
}

function ResumeSectionHeading({ title }) {
  return (
    <div style={{
      fontSize: 11,
      fontWeight: 800,
      letterSpacing: '0.16em',
      textTransform: 'uppercase',
      color: '#16466F',
      paddingBottom: 7,
      borderBottom: '1px solid rgba(22,70,111,0.18)',
    }}>
      {title}
    </div>
  );
}

function ResumeBulletList({ bullets }) {
  return (
    <div style={{ display: 'grid', gap: 7 }}>
      {asArray(bullets).filter(Boolean).map((bullet, index) => (
        <div key={index} style={{ display: 'grid', gridTemplateColumns: '14px minmax(0, 1fr)', gap: 8, alignItems: 'start' }}>
          <span style={{ fontSize: 13, lineHeight: 1.7, color: '#16466F' }}>-</span>
          <span style={{ fontSize: 13.5, lineHeight: 1.72, color: '#1F1F1C' }}>{bullet}</span>
        </div>
      ))}
    </div>
  );
}

function ResumePreviewLink({ label, url }) {
  if (!url) return null;
  return (
    <a
      href={url}
      target="_blank"
      rel="noreferrer"
      style={{
        fontSize: 11.5,
        fontWeight: 700,
        letterSpacing: '0.02em',
        color: '#185FA5',
        textDecoration: 'none',
        cursor: 'pointer',
        borderBottom: '1px solid #185FA5',
        transition: 'color 0.2s',
      }}
      onMouseEnter={(e) => e.target.style.color = '#0C447C'}
      onMouseLeave={(e) => e.target.style.color = '#185FA5'}
    >
      {label}
    </a>
  );
}

function ResumeEntryFrame({ children }) {
  return (
    <div style={{
      display: 'grid',
      gap: 7,
      paddingLeft: 14,
      borderLeft: '2px solid rgba(24,95,165,0.16)',
    }}>
      {children}
    </div>
  );
}

function ResumePreviewSection({ section }) {
  const kind = section?.kind;
  const title = section?.title;
  const lines = asArray(section?.lines).filter(Boolean);

  if (!title) return null;

  if (kind === 'summary') {
    return (
      <section style={{ display: 'grid', gap: 11 }}>
        <ResumeSectionHeading title={title} />
        <div style={{ display: 'grid', gap: 8 }}>
          {asArray(section?.paragraphs).filter(Boolean).map((paragraph, index) => (
            <p key={index} style={{ fontSize: 13.6, lineHeight: 1.78, color: '#1F1F1C' }}>{paragraph}</p>
          ))}
        </div>
      </section>
    );
  }

  if (kind === 'skills') {
    return (
      <section style={{ display: 'grid', gap: 11 }}>
        <ResumeSectionHeading title={title} />
        <div style={{ display: 'grid', gap: 10 }}>
          {asArray(section?.categories).map((category, index) => (
            <div key={index} style={{ display: 'grid', gridTemplateColumns: '110px minmax(0, 1fr)', gap: 12, alignItems: 'start' }}>
              <div style={{ fontSize: 11, fontWeight: 800, letterSpacing: '0.08em', textTransform: 'uppercase', color: '#16466F', paddingTop: 2 }}>
                {category?.label}
              </div>
              <div style={{ fontSize: 13.5, lineHeight: 1.72, color: '#1F1F1C' }}>
                {asArray(category?.items).filter(Boolean).join(', ')}
              </div>
            </div>
          ))}
        </div>
      </section>
    );
  }

  if (kind === 'experience') {
    return (
      <section style={{ display: 'grid', gap: 13 }}>
        <ResumeSectionHeading title={title} />
        <div style={{ display: 'grid', gap: 16 }}>
          {asArray(section?.items).map((item, index) => (
            <ResumeEntryFrame key={index}>
              <div style={{ fontSize: 15.5, fontWeight: 800, color: '#171714' }}>{item?.title}</div>
              <div style={{ fontSize: 12.2, lineHeight: 1.6, color: '#5A5A54' }}>
                {[item?.organization, item?.location, item?.date_range].filter(Boolean).join(' | ')}
              </div>
              <ResumePreviewLink label={item?.url_label || 'Company website'} url={item?.url} />
              <ResumeBulletList bullets={item?.bullets} />
            </ResumeEntryFrame>
          ))}
        </div>
      </section>
    );
  }

  if (kind === 'projects') {
    return (
      <section style={{ display: 'grid', gap: 13 }}>
        <ResumeSectionHeading title={title} />
        <div style={{ display: 'grid', gap: 16 }}>
          {asArray(section?.items).map((item, index) => (
            <ResumeEntryFrame key={index}>
              <div style={{ display: 'grid', gap: 3 }}>
                <div style={{ fontSize: 15.5, fontWeight: 800, color: '#171714' }}>{item?.name}</div>
                {item?.subtitle && <div style={{ fontSize: 12.6, fontWeight: 600, color: '#16466F' }}>{item.subtitle}</div>}
              </div>
              {asArray(item?.tech_stack).length > 0 && (
                <div style={{ fontSize: 12.2, lineHeight: 1.65, color: '#5A5A54' }}>
                  <span style={{ fontWeight: 700, color: '#3A3A35' }}>Tech Stack:</span> {asArray(item.tech_stack).join(', ')}
                </div>
              )}
              <ResumePreviewLink label={item?.url_label || 'Project URL'} url={item?.url} />
              <ResumeBulletList bullets={item?.bullets} />
            </ResumeEntryFrame>
          ))}
        </div>
      </section>
    );
  }

  if (kind === 'education') {
    return (
      <section style={{ display: 'grid', gap: 11 }}>
        <ResumeSectionHeading title={title} />
        <div style={{ display: 'grid', gap: 12 }}>
          {asArray(section?.items).map((item, index) => (
            <ResumeEntryFrame key={index}>
              <div style={{ fontSize: 14.6, fontWeight: 800, color: '#171714' }}>
                {[item?.institution, item?.date_range].filter(Boolean).join(' | ')}
              </div>
              <div style={{ fontSize: 12.6, lineHeight: 1.65, color: '#5A5A54' }}>
                {[item?.degree, item?.details].filter(Boolean).join(' | ')}
              </div>
            </ResumeEntryFrame>
          ))}
        </div>
      </section>
    );
  }

  if (kind === 'certifications') {
    return (
      <section style={{ display: 'grid', gap: 11 }}>
        <ResumeSectionHeading title={title} />
        <div style={{ display: 'grid', gap: 12 }}>
          {asArray(section?.items).map((item, index) => (
            <ResumeEntryFrame key={index}>
              <div style={{ fontSize: 14.6, fontWeight: 800, color: '#171714' }}>
                {[item?.name, item?.issuer, item?.date].filter(Boolean).join(' | ')}
              </div>
              <ResumePreviewLink label={item?.url_label || 'Credential'} url={item?.url} />
            </ResumeEntryFrame>
          ))}
        </div>
      </section>
    );
  }

  if (!lines.length) return null;

  return (
    <section style={{ display: 'grid', gap: 10 }}>
      <ResumeSectionHeading title={title} />
      {section.layout === 'paragraphs' ? (
        <div style={{ display: 'grid', gap: 8 }}>
          {lines.map((line, index) => (
            <p key={`${title}-${index}`} style={{ fontSize: 13.5, lineHeight: 1.72, color: '#1F1F1C' }}>
              {line}
            </p>
          ))}
        </div>
      ) : section.layout === 'bullets' || section.layout === 'highlights' ? (
        <ResumeBulletList bullets={lines} />
      ) : (
        <div style={{ display: 'grid', gap: 8 }}>
          {lines.map((line, index) => (
            <div key={`${title}-${index}`} style={{ fontSize: 13.5, lineHeight: 1.72, color: '#1F1F1C' }}>
              {line}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function ATSResumePreview({ opt }) {
  const resume = opt?.structured_resume;
  const sections = asArray(resume?.sections);
  if (!sections.length) {
    return (
      <pre style={{
        fontFamily: "'DM Mono', 'Fira Code', monospace", fontSize: 12, lineHeight: 1.75,
        background: T.bg1, border: `0.5px solid ${T.border2}`, borderRadius: 10,
        padding: '16px 18px', whiteSpace: 'pre-wrap', wordBreak: 'break-word',
        color: T.text1, maxHeight: 460, overflowY: 'auto',
      }}>{opt?.draft}</pre>
    );
  }

  const name = resume?.name;
  const contact = asArray(resume?.contact_items || resume?.contact_lines).filter(Boolean);
  const profileLinks = asArray(resume?.profile_links).filter((link) => link?.url);

  return (
    <div style={{
      background: '#FFFFFF',
      border: `1px solid ${T.border2}`,
      borderRadius: 12,
      padding: '32px 34px',
      display: 'grid',
      gap: 22,
      boxShadow: '0 14px 40px rgba(17, 31, 51, 0.06)',
    }}>
      <div style={{ display: 'grid', gap: 8 }}>
        {name && (
          <div style={{
            fontSize: 29,
            fontWeight: 700,
            letterSpacing: '0.06em',
            textTransform: 'uppercase',
            color: '#171714',
            whiteSpace: 'pre-wrap',
          }}>
            {name}
          </div>
        )}
        {contact.length > 0 && (
          <div style={{
            fontSize: 12.5,
            lineHeight: 1.7,
            color: '#55554F',
            wordBreak: 'break-word',
          }}>
            {contact.join(' | ')}
          </div>
        )}
        {profileLinks.length > 0 && (
          <div style={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: 12,
            fontSize: 12.5,
            lineHeight: 1.7,
          }}>
            {profileLinks.map((link, idx) => (
              <span key={`${link.label}-${link.url}`}>
                {idx > 0 && <span style={{ color: '#55554F' }}> | </span>}
                <ResumePreviewLink label={link.label} url={link.url} />
              </span>
            ))}
          </div>
        )}
      </div>

      {sections.map((section) => (
        <ResumePreviewSection key={section.title || JSON.stringify(section)} section={section} />
      ))}
    </div>
  );
}

function OptimisedResume({ opt, onCopy, onDownload, copied, downloadingPdf }) {
  if (!opt) return null;
  return (
    <Card>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 10, marginBottom: 16 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <SectionLabel
            title="AI-enhanced draft"
            icon={opt.can_generate
              ? <svg width="14" height="14" viewBox="0 0 16 16" fill="none"><path d="M8 2C6 5 5 8 6 11l3-1 1-3C12 4 13 2 8 2z" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round" /><path d="M6 11l-3 1 1-3" stroke="currentColor" strokeWidth="1.1" strokeLinecap="round" strokeLinejoin="round" /></svg>
              : <svg width="14" height="14" viewBox="0 0 16 16" fill="none"><rect x="3" y="7" width="10" height="8" rx="1.5" stroke="currentColor" strokeWidth="1.2" /><path d="M5 7V5a3 3 0 016 0v2" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" /></svg>
            }
          />
          {opt.industry_specific && <Pill label={opt.industry_specific.template_name} variant="purple" />}
        </div>
        {opt.can_generate && (
          <div style={{ display: 'flex', gap: 8 }}>
            <button onClick={onCopy} style={{
              padding: '7px 14px', fontSize: 12, fontWeight: 500, letterSpacing: '-0.01em',
              background: T.bg1, border: `0.5px solid ${T.border2}`,
              borderRadius: 8, color: T.text1, transition: 'background 0.15s',
            }}>{copied ? 'Copied!' : 'Copy'}</button>
            <button onClick={onDownload} disabled={downloadingPdf} style={{
              padding: '7px 14px', fontSize: 12, fontWeight: 500, letterSpacing: '-0.01em',
              background: T.blue, color: '#fff', border: 'none', borderRadius: 8,
              display: 'flex', alignItems: 'center', gap: 6, transition: 'background 0.15s',
              opacity: downloadingPdf ? 0.7 : 1,
            }}>
              <svg width="11" height="11" viewBox="0 0 14 14" fill="none">
                <path d="M7 1v8M4 7l3 3 3-3M2 11h10" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              {downloadingPdf ? 'Preparing PDF...' : 'Download PDF'}
            </button>
          </div>
        )}
      </div>

      {opt.reason && (
        <div style={{ fontSize: 13, color: T.text2, marginBottom: 14, lineHeight: 1.7 }}>{opt.reason}</div>
      )}

      {opt.can_generate ? (
        <>
          <ATSResumePreview opt={opt} />
          {opt.enhancement_summary && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(110px, 1fr))', gap: 10, marginTop: 14 }}>
              {[
                { label: 'Skills highlighted', value: opt.enhancement_summary.skills_highlighted },
                { label: 'Evidence lines', value: opt.enhancement_summary.evidence_lines_used },
                { label: 'Sections', value: opt.enhancement_summary.content_sections },
                { label: 'Optimisations', value: opt.enhancement_summary.industry_optimizations },
              ].map(({ label, value }) => (
                <div key={label} style={{ background: T.bg1, borderRadius: 10, padding: '11px 14px' }}>
                  <div style={{ fontSize: 11, color: T.text3, marginBottom: 3 }}>{label}</div>
                  <div style={{ fontSize: 18, fontWeight: 700, letterSpacing: '-0.03em' }}>{value ?? '—'}</div>
                </div>
              ))}
            </div>
          )}
        </>
      ) : (
        <div style={{
          background: T.redLt, border: `0.5px solid ${T.redMd}`,
          borderRadius: 10, padding: '24px 20px', textAlign: 'center',
        }}>
          <div style={{ fontSize: 13, fontWeight: 600, color: '#791F1F', marginBottom: 5 }}>Draft locked</div>
          <div style={{ fontSize: 13, color: T.red, lineHeight: 1.65 }}>
            Address the critical gaps above to unlock resume generation.
          </div>
        </div>
      )}

      {asArray(opt?.integrity_rules).length > 0 && (
        <div style={{ marginTop: 14, display: 'grid', gap: 5 }}>
          {asArray(opt.integrity_rules).map((rule, i) => (
            <div key={i} style={{ display: 'flex', gap: 7, fontSize: 12, color: T.text2 }}>
              <svg width="11" height="11" viewBox="0 0 12 12" fill="none" style={{ marginTop: 1, flexShrink: 0 }}>
                <path d="M2 6l3 3 5-5" stroke={T.green} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              {rule}
            </div>
          ))}
        </div>
      )}
    </Card>
  );
}

/* ─────────────────────────────────────────────
   Sidebar / input panel
───────────────────────────────────────────── */
function Sidebar({ jd, setJd, file, setFile, loading, onAnalyse, err, progressSteps }) {
  const fileRef = useRef();
  const words = jd.trim() ? jd.trim().split(/\s+/).length : 0;
  const ext = file?.name?.split('.').pop()?.toUpperCase() || null;
  const ready = !loading && jd.trim() && file;
  const active = activeProgressStep(progressSteps || buildProgressSteps());

  const onFile = (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    if (f.size > 5 * 1024 * 1024) { alert('File must be under 5 MB.'); return; }
    setFile(f);
  };

  return (
    <aside style={{ position: 'sticky', top: 72 }}>
      <div style={{
        background: T.bg0, border: `0.5px solid ${T.border2}`,
        borderRadius: 16, padding: '22px 22px 24px', overflow: 'hidden',
      }}>
        {/* Header */}
        <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase', color: T.blue, marginBottom: 5 }}>
          Configure
        </div>
        <div style={{ fontSize: 20, fontWeight: 700, letterSpacing: '-0.03em', marginBottom: 3 }}>Analyse your resume</div>
        <div style={{ fontSize: 13, color: T.text2, marginBottom: 22 }}>Match against any job description</div>

        {/* JD */}
        <label style={{ display: 'block', fontSize: 12, fontWeight: 500, marginBottom: 6 }}>Job description</label>
        <div style={{ position: 'relative' }}>
          <textarea
            value={jd}
            onChange={e => setJd(e.target.value)}
            placeholder="Paste the full job description, including requirements and responsibilities…"
            style={{
              width: '100%', minHeight: 180, resize: 'vertical',
              fontSize: 13, lineHeight: 1.65, padding: '10px 12px',
              background: T.bg1, border: `0.5px solid ${T.border2}`,
              borderRadius: 10, color: T.text1,
              boxSizing: 'border-box', transition: 'border-color 0.15s, box-shadow 0.15s',
            }}
          />
          <span style={{
            position: 'absolute', bottom: 10, right: 12,
            fontSize: 10, color: T.text3, pointerEvents: 'none',
          }}>{words}w</span>
        </div>

        {/* Stats row */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, margin: '12px 0 18px' }}>
          <MetricMini label="Words" value={words} sub="in description" />
          <MetricMini
            label="Format"
            value={ext || '—'}
            sub={file ? (file.name.length > 18 ? file.name.slice(0, 18) + '…' : file.name) : 'awaiting file'}
          />
        </div>

        {/* File drop */}
        <label style={{ display: 'block', fontSize: 12, fontWeight: 500, marginBottom: 6 }}>Resume file</label>
        <div
          onClick={() => fileRef.current?.click()}
          style={{
            border: `1px dashed ${file ? T.greenMd : T.border2}`,
            borderRadius: 10,
            background: file ? T.greenLt : T.bg1,
            padding: '18px 14px', textAlign: 'center', cursor: 'pointer',
            transition: 'all 0.2s',
          }}
        >
          <input ref={fileRef} type="file" accept=".pdf,.txt,.docx" onChange={onFile} />
          <div style={{ marginBottom: 7 }}>
            {file
              ? <svg width="22" height="22" viewBox="0 0 24 24" fill="none" style={{ display: 'block', margin: '0 auto' }}><path d="M9 12l2 2 4-4" stroke="#0F6E56" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /><path d="M3 7a2 2 0 012-2h14a2 2 0 012 2v10a2 2 0 01-2 2H5a2 2 0 01-2-2V7z" stroke="#0F6E56" strokeWidth="1.5" /></svg>
              : <svg width="22" height="22" viewBox="0 0 24 24" fill="none" style={{ display: 'block', margin: '0 auto' }}><path d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M12 4v12M8 8l4-4 4 4" stroke={T.text2} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>
            }
          </div>
          <div style={{ fontSize: 13, fontWeight: 500, color: file ? '#085041' : T.text1 }}>
            {file ? file.name : 'Drop file or click to browse'}
          </div>
          <div style={{ fontSize: 11, color: T.text2, marginTop: 2 }}>
            {file ? `${(file.size / 1024).toFixed(0)} KB` : 'PDF · DOCX · TXT · max 5 MB'}
          </div>
        </div>

        {/* Error */}
        {err && (
          <div style={{
            marginTop: 12, padding: '10px 13px', borderRadius: 10,
            background: T.redLt, border: `0.5px solid ${T.redMd}`,
            fontSize: 12, color: T.red, display: 'flex', gap: 8, alignItems: 'flex-start',
          }}>
            <svg width="14" height="14" viewBox="0 0 16 16" fill="none" style={{ marginTop: 1, flexShrink: 0 }}>
              <circle cx="8" cy="8" r="7" stroke={T.red} strokeWidth="1.5" />
              <path d="M8 5v3.5M8 11h.01" stroke={T.red} strokeWidth="1.5" strokeLinecap="round" />
            </svg>
            {err}
          </div>
        )}

        {/* CTA */}
        <button
          onClick={onAnalyse}
          disabled={!ready}
          style={{
            marginTop: 20, width: '100%', height: 44,
            background: ready ? T.blue : T.bg2,
            color: ready ? '#fff' : T.text3,
            border: 'none', borderRadius: 10,
            fontSize: 14, fontWeight: 600, letterSpacing: '-0.01em',
            display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
            transition: 'background 0.2s, transform 0.15s',
          }}
          onMouseEnter={e => ready && (e.target.style.background = '#0d4d87')}
          onMouseLeave={e => ready && (e.target.style.background = T.blue)}
        >
          {loading
            ? <><Spinner size={15} color="#fff" /> {active?.label || 'Analysing...'}</>
            : <><svg width="14" height="14" viewBox="0 0 16 16" fill="none"><path d="M8 1l1.5 5H15l-4.5 3 1.5 5L8 11l-4 3 1.5-5L1 6h5.5L8 1z" fill="currentColor" /></svg> Analyse resume</>
          }
        </button>
      </div>
    </aside>
  );
}

/* ─────────────────────────────────────────────
   Main App
───────────────────────────────────────────── */
export default function App() {
  useGlobalStyles();

  const [jd, setJd] = useState('');
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [progressSteps, setProgressSteps] = useState(() => buildProgressSteps());
  const [err, setErr] = useState('');
  const [copied, setCopied] = useState(false);
  const [downloadingPdf, setDownloadingPdf] = useState(false);

  const analyse = async () => {
    setErr(''); setResult(null); setLoading(true); setProgressSteps(buildProgressSteps());
    const fd = new FormData();
    fd.append('job_description', jd);
    fd.append('file', file);
    try {
      const apiUrl = `${API_BASE_URL}/api/optimize/stream`;
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
      try {
        const res = await fetch(apiUrl, { method: 'POST', body: fd, signal: controller.signal });
        if (!res.ok) {
          const d = await res.json().catch(() => ({}));
          throw new Error(d.detail || `Error ${res.status}`);
        }
        const data = await readProgressStream(res, (event) => {
          setProgressSteps((prev) => mergeProgressEvent(prev, event));
        });
        if (!data) throw new Error('Server finished without returning a report.');
        if (data) data.score = Math.round(Number(data.score) || 0);
        setResult(data);
      } finally {
        clearTimeout(timeoutId);
      }
    } catch (e) {
      if (e.name === 'AbortError') {
        setErr('Request timeout. Server took too long to respond.');
      } else {
        setErr(e.message || 'Connection failed. Is the server running?');
      }
    } finally {
      setLoading(false);
    }
  };

  const download = async () => {
    const optimizedResume = result?.optimized_resume;
    const draft = optimizedResume?.draft;
    if (!draft) return;

    setDownloadingPdf(true);
    try {
      const res = await fetch(`${API_BASE_URL}/api/resume/pdf`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ optimized_resume: optimizedResume }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || 'PDF download failed.');
      }

      const blob = await res.blob();
      const fileName =
        filenameFromDisposition(res.headers.get('Content-Disposition')) ||
        optimizedResume.download_pdf_filename ||
        'optimized_resume.pdf';

      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = fileName;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setErr(e.message || 'PDF download failed.');
    } finally {
      setDownloadingPdf(false);
    }
  };

  const copy = async () => {
    const draft = result?.optimized_resume?.draft;
    if (!draft) return;
    await navigator.clipboard.writeText(draft).catch(() => { });
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const matched = asArray(result?.ats_signals?.matched_terms);
  const missing = asArray(result?.ats_signals?.critical_missing_terms || result?.ats_signals?.missing_terms);
  const evidence = [...asArray(result?.ats_signals?.exact_evidence), ...asArray(result?.ats_signals?.retrieved_evidence)].filter(evidenceText);
  const breakdown = asArray(result?.score_breakdown);
  const llm = asArray(result?.ai_insight?.breakdown);
  const originalResumeText = result?.original_resume_text || result?.text || '';
  const optimizedDraft = result?.optimized_resume?.draft || '';
  // Add recruiter verdict fields at top level for easy access
  const recruiterVerdict = result?.recruiter_verdict;
  const weightedScore = result?.weighted_score;
  const confidenceScore = result?.confidence_score;
  const seniorityInference = result?.seniority_inference;

  return (
    <div style={{ minHeight: '100vh', background: '#F7F6F2', fontFamily: "'DM Sans', system-ui, sans-serif" }}>

      {/* ── Topnav ── */}
      <header style={{
        position: 'sticky', top: 0, zIndex: 50,
        background: 'rgba(255,255,255,0.92)',
        backdropFilter: 'blur(12px)',
        WebkitBackdropFilter: 'blur(12px)',
        borderBottom: `0.5px solid ${T.border2}`,
        padding: '0 24px', height: 56,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            width: 30, height: 30, borderRadius: 8, background: T.blue,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
              <circle cx="7" cy="7" r="5.5" stroke="white" strokeWidth="1.6" />
              <circle cx="7" cy="7" r="2" fill="white" />
            </svg>
          </div>
          <span style={{ fontSize: 15, fontWeight: 700, letterSpacing: '-0.03em' }}>Aura.ai</span>
          <span style={{
            fontSize: 10, fontWeight: 600, letterSpacing: '0.04em',
            color: T.blue, background: T.blueLt,
            border: `0.5px solid ${T.blueMd}`,
            borderRadius: 99, padding: '2px 9px',
          }}>Resume Signal Lab</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <div style={{ width: 6, height: 6, borderRadius: '50%', background: T.green }} />
          <span style={{ fontSize: 12, color: T.text2 }}>Engine ready</span>
        </div>
      </header>

      {/* ── Layout ── */}
      <div style={{
        maxWidth: 1320, margin: '0 auto', padding: '24px 20px',
        display: 'grid',
        gridTemplateColumns: 'minmax(0, 300px) minmax(0, 1fr)',
        gap: 20, alignItems: 'start',
      }}>
        <Sidebar
          jd={jd} setJd={setJd}
          file={file} setFile={setFile}
          loading={loading} onAnalyse={analyse}
          err={err} progressSteps={progressSteps}
        />

        <main style={{ display: 'grid', gap: 14 }}>
          {loading ? (
            <LoadingState progressSteps={progressSteps} />
          ) : result ? (
            <>
              <ScoreHero result={result} />
              <ScoreBreakdown breakdown={breakdown} />
              <TwoCol left={<AISummary result={result} warnings={result.pipeline_warnings} />} right={<MatchedKeywords matched={matched} />} />
              <TwoCol left={<MissingKeywords missing={missing} />} right={<ResumeEvidence evidence={evidence} />} />
              <StrategicRecs ag={result.aggressive_feedback} />
              <DetailedReview llm={llm} />
              <ResumeDiff original={originalResumeText} draft={optimizedDraft} />
              <OptimisedResume
                opt={result.optimized_resume}
                onCopy={copy}
                onDownload={download}
                copied={copied}
                downloadingPdf={downloadingPdf}
              />
            </>
          ) : (
            <EmptyState />
          )}
        </main>
      </div>
    </div>
  );
}
