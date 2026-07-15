import { useMemo, useRef, useState, useEffect } from "react";
import { dataset } from "./dataset.js";
import { model, predict, trainMetrics, testMetrics } from "./model.js";

const CATEGORIES = [
  { min: 0.8, label: "Very High Human Development", color: "#0e9f6e" },
  { min: 0.7, label: "High Human Development", color: "#1ca5d8" },
  { min: 0.55, label: "Medium Human Development", color: "#e0a800" },
  { min: 0, label: "Low Human Development", color: "#e0524d" },
];

function categoryFor(score) {
  return CATEGORIES.find((c) => score >= c.min);
}

function summaryFor(country, score) {
  const cat = categoryFor(score);
  const adj =
    score >= 0.8
      ? "excellent"
      : score >= 0.7
      ? "strong"
      : score >= 0.55
      ? "moderate"
      : "developing";
  const income =
    score >= 0.8 ? "high national income" : score >= 0.7 ? "solid national income" : score >= 0.55 ? "growing national income" : "low national income";
  return `${country} demonstrates ${cat.label.toLowerCase()} with ${adj} education, a life expectancy that reflects ${adj} health outcomes, and ${income}.`;
}

const formatGni = (v) =>
  "$" + Math.round(v).toLocaleString("en-US");

export default function App() {
  const countries = useMemo(
    () => dataset.map((r) => r.country).sort((a, b) => a.localeCompare(b)),
    []
  );
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [selected, setSelected] = useState(null);
  const [result, setResult] = useState(null);
  const [predicting, setPredicting] = useState(false);
  const dropdownRef = useRef(null);
  const listRef = useRef(null);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return countries;
    return countries.filter((c) => c.toLowerCase().includes(q));
  }, [query, countries]);

  useEffect(() => {
    function onClick(e) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, []);

  function selectCountry(name) {
    setSelected(name);
    setOpen(false);
    setQuery("");
    setResult(null);
  }

  function handlePredict() {
    if (!selected) return;
    setPredicting(true);
    const row = dataset.find((r) => r.country === selected);
    setTimeout(() => {
      const score = predict(model, row);
      setResult({ score, country: selected, row });
      setPredicting(false);
    }, 450);
  }

  const indicators = selected
    ? dataset.find((r) => r.country === selected)
    : null;

  return (
    <div className="app">
      <BackgroundDecor />
      <main className="container">
        <Header />

        <section className="card predictor">
          <h2 className="section-title">Select a Country</h2>
          <div className="dropdown" ref={dropdownRef}>
            <button
              className={`dropdown-trigger ${open ? "open" : ""}`}
              onClick={() => setOpen((o) => !o)}
              aria-haspopup="listbox"
              aria-expanded={open}
            >
              <span className={selected ? "trigger-value" : "trigger-placeholder"}>
                {selected || "Search for a country…"}
              </span>
              <svg className="chevron" width="16" height="16" viewBox="0 0 24 24" fill="none">
                <path d="M6 9l6 6 6-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </button>
            {open && (
              <div className="dropdown-panel">
                <div className="search-wrap">
                  <svg className="search-icon" width="16" height="16" viewBox="0 0 24 24" fill="none">
                    <circle cx="11" cy="11" r="7" stroke="currentColor" strokeWidth="2" />
                    <path d="M21 21l-4-4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                  </svg>
                  <input
                    autoFocus
                    className="search-input"
                    placeholder="Type a country name…"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                  />
                </div>
                <ul className="country-list" ref={listRef}>
                  {filtered.length === 0 && (
                    <li className="no-results">No countries found</li>
                  )}
                  {filtered.slice(0, 200).map((c) => (
                    <li
                      key={c}
                      className={`country-item ${c === selected ? "active" : ""}`}
                      onClick={() => selectCountry(c)}
                    >
                      {c}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          <div className="indicators">
            <Indicator label="Life Expectancy" value={indicators ? `${indicators.lifeExpectancy.toFixed(1)} yrs` : "—"} icon="heart" />
            <Indicator label="Mean Years of Schooling" value={indicators ? `${indicators.meanYearsSchooling.toFixed(1)} yrs` : "—"} icon="book" />
            <Indicator label="Expected Years of Schooling" value={indicators ? `${indicators.expectedYearsSchooling.toFixed(1)} yrs` : "—"} icon="cap" />
            <Indicator label="GNI Per Capita" value={indicators ? formatGni(indicators.gniPerCapita) : "—"} icon="coin" />
          </div>

          <button
            className="predict-btn"
            onClick={handlePredict}
            disabled={!selected || predicting}
          >
            {predicting ? (
              <>
                <span className="spinner" /> Predicting…
              </>
            ) : (
              "Predict HDI"
            )}
          </button>
        </section>

        {result && <Result result={result} />}

        {!result && <ModelInfo />}
      </main>
      <Footer />
    </div>
  );
}

function Header() {
  return (
    <header className="header">
      <div className="eyebrow">A Comprehensive Measure of Well-Being</div>
      <h1 className="title">Human Development Index Predictor</h1>
      <p className="subtitle">
        Select any country to instantly retrieve its development indicators and
        predict its HDI score using a trained linear regression model.
      </p>
    </header>
  );
}

function Indicator({ label, value, icon }) {
  const paths = {
    heart: "M12 21s-7-4.5-9.5-9C1 9 2.5 5.5 6 5.5c2 0 3.5 1.5 4 2.5.5-1 2-2.5 4-2.5 3.5 0 5 3.5 3.5 6.5C19 16.5 12 21 12 21z",
    book: "M4 4h7a3 3 0 013 3v13a2.5 2.5 0 00-2.5-2.5H4V4zm16 0h-7a3 3 0 00-3 3v13a2.5 2.5 0 012.5-2.5H20V4z",
    cap: "M3 9l9-4 9 4-9 4-9-4zm5 2.5V16a4 4 0 008 0v-4.5",
    coin: "M12 2a10 10 0 100 20 10 10 0 000-20zm0 4a6 6 0 100 12 6 6 0 000-12z",
  };
  return (
    <div className="indicator">
      <div className="indicator-icon">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
          <path d={paths[icon]} stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </div>
      <div className="indicator-body">
        <div className="indicator-label">{label}</div>
        <div className="indicator-value">{value}</div>
      </div>
    </div>
  );
}

function Result({ result }) {
  const { score, country, row } = result;
  const cat = categoryFor(score);
  const pct = Math.round(score * 1000) / 10;
  const actual = row.hdi;
  const diff = Math.abs(score - actual);

  return (
    <section className="card result-card reveal">
      <div className="result-header">
        <span className="result-label">Predicted HDI</span>
        <span className="result-country">{country}</span>
      </div>
      <div className="result-score-row">
        <div className="result-score" style={{ color: cat.color }}>
          {score.toFixed(3)}
        </div>
        <div className="result-bar-wrap">
          <div className="result-bar-track">
            <div
              className="result-bar-fill"
              style={{ width: `${Math.min(score, 1) * 100}%`, background: cat.color }}
            />
          </div>
          <div className="result-bar-scale">
            <span>0.0</span><span>0.55</span><span>0.70</span><span>0.80</span><span>1.0</span>
          </div>
        </div>
      </div>
      <div className="result-category" style={{ background: cat.color + "1a", color: cat.color, borderColor: cat.color + "40" }}>
        {cat.label}
      </div>
      <p className="result-summary">{summaryFor(country, score)}</p>
      <div className="result-compare">
        <div className="compare-item">
          <span className="compare-label">Actual HDI (dataset)</span>
          <span className="compare-value">{actual.toFixed(3)}</span>
        </div>
        <div className="compare-item">
          <span className="compare-label">Prediction Error</span>
          <span className="compare-value">{diff.toFixed(3)}</span>
        </div>
        <div className="compare-item">
          <span className="compare-label">Percentile</span>
          <span className="compare-value">{pct}%</span>
        </div>
      </div>
    </section>
  );
}

function ModelInfo() {
  return (
    <section className="card model-info">
      <h3 className="section-title">Model Performance</h3>
      <p className="model-desc">
        A linear regression model trained on {dataset.length} countries using life
        expectancy, mean and expected years of schooling, and log GNI per capita
        as features.
      </p>
      <div className="metrics">
        <Metric label="Test R²" value={testMetrics.r2.toFixed(4)} hint="Variance explained" />
        <Metric label="Test RMSE" value={testMetrics.rmse.toFixed(4)} hint="Root mean squared error" />
        <Metric label="Test MAPE" value={testMetrics.mape.toFixed(2) + "%"} hint="Mean abs % error" />
        <Metric label="Train R²" value={trainMetrics.r2.toFixed(4)} hint="Training fit" />
      </div>
    </section>
  );
}

function Metric({ label, value, hint }) {
  return (
    <div className="metric">
      <div className="metric-value">{value}</div>
      <div className="metric-label">{label}</div>
      <div className="metric-hint">{hint}</div>
    </div>
  );
}

function BackgroundDecor() {
  return (
    <div className="bg-decor" aria-hidden="true">
      <div className="blob blob-1" />
      <div className="blob blob-2" />
      <div className="grid-overlay" />
    </div>
  );
}

function Footer() {
  return (
    <footer className="footer">
      <span>HDI Prediction System · Linear Regression · {dataset.length} countries</span>
    </footer>
  );
}
