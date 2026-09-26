// React is provided by the dashboard's runtime; keep type-checking this standalone file
// even when the local TypeScript environment has not resolved React's declarations.
// @ts-ignore -- dependency resolution is handled by the dashboard build configuration.
import React, { useCallback, useState } from "react";

// --- Interfaces ---
export interface Decision {
  bank: string;
  card: string;
  recommended: boolean;
  reason: string;
  cited_chunk_id: string | null;
}

export interface RecommendOut {
  decisions: Decision[];
  latency_ms: number;
  retrieved_k: number;
}

export interface ProfileForm {
  annual_income: number;
  age: number;
  cibil: number;
  employment: "salaried" | "self-employed" | "business owner";
  existing_relationship: boolean;
  banks: string[];
}

// --- Constants ---
const DEFAULT_PROFILE: ProfileForm = {
  annual_income: 800000,
  age: 27,
  cibil: 760,
  employment: "salaried",
  existing_relationship: false,
  banks: ["SBI", "HDFC"],
};

// Replace this with your actual environment variable setup or hardcoded URL
const API_BASE = import.meta.env?.VITE_API_BASE ?? "http://localhost:8000";

// --- Component ---
export default function RecommenderPanel(): JSX.Element {
  const [profile, setProfile] = useState<ProfileForm>(DEFAULT_PROFILE);
  const [result, setResult] = useState<RecommendOut | null>(null);
  const [p95, setP95] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refreshTelemetry = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/telemetry`);
      const data = await res.json();
      setP95(data.p95_latency_ms as number);
    } catch {
      // telemetry is best-effort
    }
  }, []);

  const handleSubmit = useCallback(
    async (event: React.FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      setError(null);
      try {
        const res = await fetch(`${API_BASE}/recommend`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(profile),
        });
        
        if (!res.ok) {
          throw new Error(`Request failed with status ${res.status}`);
        }
        
        const data = (await res.json()) as RecommendOut;
        setResult(data);
        await refreshTelemetry();
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      }
    },
    [profile, refreshTelemetry]
  );

  return (
    <div className="recommender-panel">
      <h1>Eligibility-Aware Credit Card Recommender</h1>

      <form onSubmit={handleSubmit}>
        <label>
          Annual income (INR)
          <input
            type="number"
            min={0}
            value={profile.annual_income}
            onChange={(e) =>
              setProfile({ ...profile, annual_income: Number(e.target.value) })
            }
          />
        </label>

        <label>
          Age
          <input
            type="number"
            min={18}
            max={100}
            value={profile.age}
            onChange={(e) =>
              setProfile({ ...profile, age: Number(e.target.value) })
            }
          />
        </label>

        <label>
          CIBIL score
          <input
            type="number"
            min={300}
            max={900}
            value={profile.cibil}
            onChange={(e) =>
              setProfile({ ...profile, cibil: Number(e.target.value) })
            }
          />
        </label>

        <label>
          Employment
          <select
            value={profile.employment}
            onChange={(e) =>
              setProfile({
                ...profile,
                employment: e.target.value as ProfileForm["employment"],
              })
            }
          >
            <option value="salaried">Salaried</option>
            <option value="self-employed">Self-employed</option>
            <option value="business owner">Business owner</option>
          </select>
        </label>

        <label>
          Existing bank relationship
          <select
            value={profile.existing_relationship ? "true" : "false"}
            onChange={(e) =>
              setProfile({
                ...profile,
                existing_relationship: e.target.value === "true",
              })
            }
          >
            <option value="false">No</option>
            <option value="true">Yes</option>
          </select>
        </label>

        <button type="submit">Get recommendations</button>
      </form>

      {error && <div className="error">Error: {error}</div>}

      {result && (
        <div className="stats">
          <div className="stat">
            <div className="value">{result.latency_ms.toFixed(1)} ms</div>
            <div className="label">Latency (L_ms)</div>
          </div>
          <div className="stat">
            <div className="value">{result.retrieved_k}</div>
            <div className="label">Retrieved k</div>
          </div>
          <div className="stat">
            <div className="value">{result.decisions.length}</div>
            <div className="label">Decisions</div>
          </div>
          <div className="stat">
            <div className="value">
              {p95 !== null ? `${p95.toFixed(0)} ms` : "-"}
            </div>
            <div className="label">p95 latency (telemetry)</div>
          </div>
        </div>
      )}

      {result && (
        <div className="cards">
          {result.decisions.map((decision) => (
            <div
              key={`${decision.bank}-${decision.card}`}
              className={`card ${
                decision.recommended ? "recommended" : "not-eligible"
              }`}
            >
              <span className="badge">
                {decision.recommended ? "RECOMMENDED" : "NOT ELIGIBLE"}
              </span>
              <h3>
                {decision.bank} - {decision.card}
              </h3>
              <p>{decision.reason}</p>
              <p className="chunk">
                Cited chunk: {decision.cited_chunk_id ?? "n/a"}
              </p>
            </div>
          ))}
        </div>
      )}

      {/* Basic styling to make the output readable */}
      <style>{`
        .recommender-panel { font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        form { display: flex; flex-direction: column; gap: 15px; margin-bottom: 20px; }
        label { display: flex; flex-direction: column; font-weight: bold; }
        input, select { padding: 8px; margin-top: 5px; border: 1px solid #ccc; border-radius: 4px; }
        button { padding: 10px; background-color: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background-color: #0056b3; }
        .error { color: red; margin-bottom: 20px; padding: 10px; background: #ffe6e6; border-radius: 4px; }
        .stats { display: flex; gap: 20px; margin-bottom: 20px; padding: 15px; background: #f8f9fa; border-radius: 8px; }
        .stat { text-align: center; }
        .stat .value { font-size: 1.2em; font-weight: bold; }
        .stat .label { font-size: 0.8em; color: #666; }
        .cards { display: flex; flex-direction: column; gap: 15px; }
        .card { padding: 15px; border: 1px solid #ddd; border-radius: 8px; position: relative; }
        .card.recommended { border-left: 5px solid #28a745; }
        .card.not-eligible { border-left: 5px solid #dc3545; opacity: 0.7; }
        .badge { font-size: 0.7em; padding: 3px 6px; border-radius: 4px; background: #eee; font-weight: bold; }
        .recommended .badge { background: #d4edda; color: #155724; }
        .not-eligible .badge { background: #f8d7da; color: #721c24; }
        .chunk { font-size: 0.8em; color: #888; font-family: monospace; }
      `}</style>
    </div>
  );
}