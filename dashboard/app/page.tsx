"use client";

import { useEffect, useMemo, useState } from "react";

type RiskRow = {
  t: string;
  rank: number;
  pitcher: string;
  name: string;
  role: "SP" | "RP" | "swing";
  P90: number;
  P150: number;
  pct: number;
  drivers: string;
  surgery_within_90d: string;
  surgery_within_150d: string;
  surgery_date: string;
  days_to_surgery: string;
};

type Driver = { label: string; value: number };

const STATE_HASH =
  "e14ba800227a5b65a12ca55114e106e20a4636857ef947d5997b9e496e02fac8";
const DATA_HASH =
  "d2f68cd38bbd5cdbb9bd5009280f5afaf534db0eea494312e95c4c083ade1228";
const DEFAULT_DATE = "2024-08-01";

function parseCsv(text: string): RiskRow[] {
  const records: string[][] = [];
  let record: string[] = [];
  let field = "";
  let quoted = false;

  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const next = text[i + 1];
    if (char === '"' && quoted && next === '"') {
      field += '"';
      i += 1;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (char === "," && !quoted) {
      record.push(field);
      field = "";
    } else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && next === "\n") i += 1;
      record.push(field);
      if (record.some((cell) => cell.length > 0)) records.push(record);
      record = [];
      field = "";
    } else {
      field += char;
    }
  }
  if (field.length > 0 || record.length > 0) {
    record.push(field);
    records.push(record);
  }

  const [header, ...body] = records;
  if (!header) return [];
  return body.map((cells) => {
    const row = Object.fromEntries(header.map((key, index) => [key, cells[index] ?? ""]));
    return {
      ...(row as Omit<RiskRow, "rank" | "P90" | "P150" | "pct">),
      rank: Number(row.rank),
      P90: Number(row.P90),
      P150: Number(row.P150),
      pct: Number(row.pct),
    } as RiskRow;
  });
}

function parseDrivers(value: string): Driver[] {
  return value
    .split(";")
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => {
      const match = part.match(/^(.*)\(([+-]?\d+(?:\.\d+)?)\)$/);
      return match
        ? { label: match[1].trim(), value: Number(match[2]) }
        : { label: part, value: 0 };
    });
}

function probability(value: number) {
  return `${(value * 100).toFixed(2)}%`;
}

function dateLabel(value: string) {
  const [year, month, day] = value.split("-");
  return `${year}.${month}.${day}`;
}

function outcomeLabel(value: string) {
  if (value === "1") return "관찰됨";
  if (value === "0") return "관찰 안 됨";
  return "라벨 미성숙";
}

export default function Home() {
  const [rows, setRows] = useState<RiskRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [date, setDate] = useState(DEFAULT_DATE);
  const [role, setRole] = useState("all");
  const [query, setQuery] = useState("");
  const [selectedKey, setSelectedKey] = useState("");

  useEffect(() => {
    fetch("/data/demo_test_top20.csv")
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.text();
      })
      .then((text) => {
        const parsed = parseCsv(text);
        if (parsed.length === 0) throw new Error("데이터가 비어 있습니다.");
        setRows(parsed);
        if (!parsed.some((row) => row.t === DEFAULT_DATE)) setDate(parsed[0].t);
      })
      .catch((reason: unknown) => {
        setError(reason instanceof Error ? reason.message : "데이터를 읽지 못했습니다.");
      })
      .finally(() => setLoading(false));
  }, []);

  const dates = useMemo(
    () => Array.from(new Set(rows.map((row) => row.t))).sort().reverse(),
    [rows],
  );

  const dateRows = useMemo(() => rows.filter((row) => row.t === date), [rows, date]);
  const visibleRows = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase();
    return dateRows.filter((row) => {
      const roleMatches = role === "all" || row.role === role;
      const queryMatches =
        needle.length === 0 ||
        row.name.toLocaleLowerCase().includes(needle) ||
        row.pitcher.includes(needle);
      return roleMatches && queryMatches;
    });
  }, [dateRows, query, role]);

  const selected =
    visibleRows.find((row) => `${row.t}-${row.pitcher}` === selectedKey) ?? visibleRows[0];
  const drivers = selected ? parseDrivers(selected.drivers) : [];
  const driverMax = Math.max(0.01, ...drivers.map((driver) => Math.abs(driver.value)));
  const topP150 = dateRows.reduce<RiskRow | undefined>(
    (best, row) => (!best || row.P150 > best.P150 ? row : best),
    undefined,
  );
  const observed150 = dateRows.filter((row) => row.surgery_within_150d === "1").length;
  const mature150 = dateRows.filter((row) => row.surgery_within_150d !== "라벨미성숙").length;
  const rpCount = dateRows.filter((row) => row.role === "RP" || row.role === "swing").length;

  return (
    <div className="site-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="PAINS 연구 대시보드 홈">
          <span className="brand-mark" aria-hidden="true">P</span>
          <span>
            <strong>PAINS</strong>
            <small>Pitcher injury signal lab</small>
          </span>
        </a>
        <div className="model-state" title="동결 모델 상태">
          <span className="status-dot" aria-hidden="true" />
          FROZEN · q0
        </div>
      </header>

      <main id="top">
        <section className="hero">
          <div>
            <p className="eyebrow">MLB TJS · RETROSPECTIVE DEMO</p>
            <h1>동결된 위험 순위의 근거를<br />한 화면에서 확인합니다.</h1>
            <p className="hero-copy">
              90일·150일 위험 점수와 순위 기여 요인을 함께 보여주는 연구용 뷰입니다.
              아래 결과는 2022–2024 과거 시점의 재현용 상위 20명 표본입니다.
            </p>
          </div>
          <div className="hero-stamp" aria-label="동결일 2026년 7월 13일">
            <span>MODEL LOCK</span>
            <strong>13 JUL</strong>
            <small>2026 · v1.0</small>
          </div>
        </section>

        <aside className="research-notice">
          <strong>연구 경보 ≠ 임상 진단</strong>
          <span>
            절대확률 보정 게이트를 통과하지 못했으므로 점수는 상대적 우선순위로만 해석합니다.
            현재 선수의 실시간 위험 명단이 아닙니다.
          </span>
        </aside>

        <section className="controls" aria-label="데이터 필터">
          <label>
            <span>결정 시점</span>
            <select value={date} onChange={(event) => { setDate(event.target.value); setSelectedKey(""); }}>
              {dates.map((item) => <option key={item} value={item}>{dateLabel(item)}</option>)}
            </select>
          </label>
          <label>
            <span>역할</span>
            <select value={role} onChange={(event) => { setRole(event.target.value); setSelectedKey(""); }}>
              <option value="all">전체 역할</option>
              <option value="SP">선발 (SP)</option>
              <option value="RP">불펜 (RP)</option>
              <option value="swing">스윙맨</option>
            </select>
          </label>
          <label className="search-field">
            <span>선수 검색</span>
            <input
              type="search"
              value={query}
              onChange={(event) => { setQuery(event.target.value); setSelectedKey(""); }}
              placeholder="이름 또는 MLBAM ID"
            />
          </label>
          <div className="data-scope">
            <span>표시 범위</span>
            <strong>TOP 20 / canonical TOP 50</strong>
          </div>
        </section>

        {loading && <div className="state-panel">검증 표본을 불러오는 중입니다…</div>}
        {error && <div className="state-panel error">데이터 로드 실패: {error}</div>}

        {!loading && !error && (
          <>
            <section className="metric-grid" aria-label="선택 시점 요약">
              <article>
                <span>최상위 P150</span>
                <strong>{topP150 ? probability(topP150.P150) : "—"}</strong>
                <small>{topP150?.name ?? "—"} · 상대 순위 1위</small>
              </article>
              <article>
                <span>150일 관찰 결과</span>
                <strong>{observed150}<em> / {mature150}</em></strong>
                <small>상위 20명 중 라벨 성숙 표본</small>
              </article>
              <article>
                <span>비선발 표본</span>
                <strong>{rpCount}<em> / {dateRows.length}</em></strong>
                <small>RP + swing · 불펜 희소성 표시</small>
              </article>
              <article className="metric-accent">
                <span>운영 정책</span>
                <strong>q0</strong>
                <small>역할별 강제 할당 없음</small>
              </article>
            </section>

            <section className="workspace">
              <div className="ranking-panel">
                <div className="section-heading">
                  <div>
                    <p className="eyebrow">RISK QUEUE</p>
                    <h2>{dateLabel(date)} 위험 우선순위</h2>
                  </div>
                  <span>{visibleRows.length}명 표시</span>
                </div>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>순위</th>
                        <th>투수</th>
                        <th>역할</th>
                        <th>P90</th>
                        <th>P150</th>
                        <th>백분위</th>
                        <th>150일 결과</th>
                      </tr>
                    </thead>
                    <tbody>
                      {visibleRows.map((row) => {
                        const key = `${row.t}-${row.pitcher}`;
                        const active = selected && selected.t === row.t && selected.pitcher === row.pitcher;
                        return (
                          <tr
                            key={key}
                            className={active ? "active" : ""}
                            onClick={() => setSelectedKey(key)}
                          >
                            <td><span className="rank">{String(row.rank).padStart(2, "0")}</span></td>
                            <td>
                              <button type="button" onClick={() => setSelectedKey(key)}>
                                <strong>{row.name}</strong>
                                <small>MLBAM {row.pitcher}</small>
                              </button>
                            </td>
                            <td><span className={`role role-${row.role}`}>{row.role}</span></td>
                            <td className="number">{probability(row.P90)}</td>
                            <td className="number strong-number">{probability(row.P150)}</td>
                            <td className="number">{row.pct.toFixed(1)}</td>
                            <td><span className={`outcome outcome-${row.surgery_within_150d}`}>{outcomeLabel(row.surgery_within_150d)}</span></td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                  {visibleRows.length === 0 && <p className="empty">조건에 맞는 선수가 없습니다.</p>}
                </div>
              </div>

              <aside className="detail-panel">
                {selected ? (
                  <>
                    <div className="detail-head">
                      <div>
                        <p className="eyebrow">SELECTED PITCHER</p>
                        <h2>{selected.name}</h2>
                        <span>MLBAM {selected.pitcher} · {selected.role} · #{selected.rank}</span>
                      </div>
                      <div className="percentile-ring" style={{ "--pct": `${selected.pct * 3.6}deg` } as React.CSSProperties}>
                        <strong>{selected.pct.toFixed(1)}</strong>
                        <small>백분위</small>
                      </div>
                    </div>

                    <div className="horizon-grid">
                      <div><span>90일 점수</span><strong>{probability(selected.P90)}</strong></div>
                      <div><span>150일 점수</span><strong>{probability(selected.P150)}</strong></div>
                    </div>

                    <div className="drivers">
                      <div className="subheading">
                        <h3>상위 기여 요인</h3>
                        <span>표준화 log-hazard 기여</span>
                      </div>
                      {drivers.map((driver) => (
                        <div className="driver" key={`${driver.label}-${driver.value}`}>
                          <div><span>{driver.label}</span><strong>{driver.value >= 0 ? "+" : ""}{driver.value.toFixed(2)}</strong></div>
                          <div className="driver-track"><i style={{ width: `${Math.max(6, Math.abs(driver.value) / driverMax * 100)}%` }} /></div>
                        </div>
                      ))}
                    </div>

                    <div className="outcome-card">
                      <h3>사후 관찰</h3>
                      <dl>
                        <div><dt>90일 TJS</dt><dd>{outcomeLabel(selected.surgery_within_90d)}</dd></div>
                        <div><dt>150일 TJS</dt><dd>{outcomeLabel(selected.surgery_within_150d)}</dd></div>
                        <div><dt>수술일</dt><dd>{selected.surgery_date || "—"}</dd></div>
                      </dl>
                      <p>이 결과 정보는 과거 재현 검증에만 표시되며 실제 전향 점수 산출에는 입력되지 않습니다.</p>
                    </div>
                  </>
                ) : <p className="empty">선수를 선택해 주세요.</p>}
              </aside>
            </section>
          </>
        )}

        <section className="method-grid">
          <article>
            <p className="eyebrow">READING GUIDE</p>
            <h2>점수는 “누가 더 먼저 검토되어야 하는가”를 뜻합니다.</h2>
            <p>
              P90과 P150은 각각 결정일 이후 90일·150일 내 TJS 위험의 동결 모델 출력입니다.
              절대확률로 진단하거나 의료 결정을 내리는 용도가 아닙니다.
            </p>
          </article>
          <article>
            <p className="eyebrow">DATA BOUNDARY</p>
            <h2>결정일 이전 정보만 사용합니다.</h2>
            <p>
              피처는 엄격한 <code>game_date &lt; t</code> 규칙을 따릅니다. 이 화면의 수술 결과는
              평가용 사후 라벨이며 모델 입력과 분리되어 있습니다.
            </p>
          </article>
          <article>
            <p className="eyebrow">BULLPEN POLICY</p>
            <h2>불펜 강제 quota는 채택하지 않았습니다.</h2>
            <p>
              q20 안전성 H150에서 기존 경보 4건을 잃어 gate를 통과하지 못했습니다.
              따라서 전체 투수 점수순 q0가 canonical 정책입니다.
            </p>
          </article>
        </section>

        <section className="integrity">
          <div>
            <p className="eyebrow">REPRODUCIBILITY</p>
            <h2>동결 상태와 데모 데이터의 무결성</h2>
          </div>
          <dl>
            <div><dt>Model state SHA-256</dt><dd title={STATE_HASH}>{STATE_HASH}</dd></div>
            <div><dt>Demo data SHA-256</dt><dd title={DATA_HASH}>{DATA_HASH}</dd></div>
            <div><dt>Decision protocol</dt><dd>Frozen v1.0 · q0 · strict as-of</dd></div>
          </dl>
        </section>
      </main>

      <footer>
        <span>PAINS · Pitcher TJS Prediction Research</span>
        <span>Retrospective demonstration · Not for clinical use</span>
      </footer>
    </div>
  );
}
