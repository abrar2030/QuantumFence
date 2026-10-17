import React, { useState, useEffect, useCallback, useMemo } from "react";
import { droneApi } from "../services/api";
import { useWebSocket } from "../context/WebSocketContext";
const RADAR_RANGE_M = 300;
function threatColor(score) {
  if (score >= 0.7) return "var(--qf-red)";
  if (score >= 0.4) return "var(--qf-orange)";
  return "var(--qf-yellow)";
}
function DroneRadar({ drones, tracks }) {
  const S = 280,
    C = S / 2;
  const blips =
    tracks.length > 0
      ? tracks.map((t, i) => {
          const hasBearing =
            typeof t.bearing_deg === "number" && typeof t.range_m === "number";
          const angle = hasBearing
            ? (t.bearing_deg * Math.PI) / 180 - Math.PI / 2
            : (i / Math.max(tracks.length, 1)) * 2 * Math.PI - Math.PI / 2;
          const dist = hasBearing
            ? Math.min(Math.max(t.range_m / RADAR_RANGE_M, 0.08), 0.95)
            : 0.3 + (t.confidence || 0.7) * 0.55;
          return {
            key: `${t.camera_id}:${t.track_id}`,
            angle,
            dist,
            color: threatColor(t.threat_score ?? 0.5),
            label: `#${t.track_id}`,
          };
        })
      : drones.slice(0, 8).map((d, i) => ({
          key: d.id || i,
          angle: (i / Math.max(drones.length, 1)) * 2 * Math.PI - Math.PI / 2,
          dist: 0.3 + (d.confidence || 0.7) * 0.55,
          color: "var(--qf-red)",
          label: `${((d.confidence || 0) * 100).toFixed(0)}%`,
        }));
  return (
    <svg
      width={S}
      height={S}
      style={{
        display: "block",
        margin: "0 auto",
      }}
    >
      <defs>
        <radialGradient id="sweep">
          <stop offset="0%" stopColor="#00ff88" stopOpacity="0.6" />
          <stop offset="100%" stopColor="#00ff88" stopOpacity="0" />
        </radialGradient>
      </defs>

      {}
      {[0.25, 0.5, 0.75, 1.0].map((r, i) => (
        <circle
          key={i}
          cx={C}
          cy={C}
          r={C * r}
          fill="none"
          stroke="var(--qf-border)"
          strokeWidth={1}
          strokeDasharray="4 4"
        />
      ))}
      {}
      <line
        x1={C}
        y1={0}
        x2={C}
        y2={S}
        stroke="var(--qf-border)"
        strokeWidth={0.5}
      />
      <line
        x1={0}
        y1={C}
        x2={S}
        y2={C}
        stroke="var(--qf-border)"
        strokeWidth={0.5}
      />

      {}
      <g
        style={{
          transformOrigin: `${C}px ${C}px`,
          animation: "rotate-ring 3s linear infinite",
        }}
      >
        <path
          d={`M${C},${C} L${C},${C - C * 0.99} A${C * 0.99},${C * 0.99} 0 0,1 ${C + C * 0.99 * Math.sin(Math.PI / 4)},${C - C * 0.99 * Math.cos(Math.PI / 4)} Z`}
          fill="url(#sweep)"
          opacity={0.35}
        />
      </g>

      {}
      <circle cx={C} cy={C} r={5} fill="var(--qf-green)" />

      {blips.map((b) => {
        const x = C + Math.cos(b.angle) * C * b.dist;
        const y = C + Math.sin(b.angle) * C * b.dist;
        return (
          <g key={b.key}>
            <circle cx={x} cy={y} r={7} fill={b.color} opacity={0.85} />
            <circle
              cx={x}
              cy={y}
              r={14}
              fill="none"
              stroke={b.color}
              strokeWidth={1}
              opacity={0}
            >
              <animate
                attributeName="r"
                from="7"
                to="22"
                dur="2s"
                repeatCount="indefinite"
              />
              <animate
                attributeName="opacity"
                from="0.5"
                to="0"
                dur="2s"
                repeatCount="indefinite"
              />
            </circle>
            <text
              x={x + 11}
              y={y - 5}
              fill={b.color}
              fontSize={9}
              fontFamily="var(--font-mono)"
            >
              {b.label}
            </text>
          </g>
        );
      })}

      {}
      {[
        ["N", C, 11],
        ["E", S - 7, C + 4],
        ["S", C, S - 5],
        ["W", 9, C + 4],
      ].map(([l, x, y]) => (
        <text
          key={l}
          x={x}
          y={y}
          textAnchor="middle"
          fill="var(--qf-text-muted)"
          fontSize={9}
          fontFamily="var(--font-mono)"
        >
          {l}
        </text>
      ))}
    </svg>
  );
}
export default function DroneWatch() {
  const [drones, setDrones] = useState([]);
  const [stats, setStats] = useState(null);
  const [liveDetections, setLiveDetections] = useState([]);
  const [loading, setLoading] = useState(true);
  const [liveMap, setLiveMap] = useState({});
  const [polledTracks, setPolledTracks] = useState([]);
  const [polledSwarm, setPolledSwarm] = useState(false);
  const { latestDrone, latestDetection, connected } = useWebSocket();
  const load = useCallback(async () => {
    try {
      const [d, s, t] = await Promise.all([
        droneApi.list({
          hours: 2,
          limit: 50,
        }),
        droneApi.stats(),
        droneApi.tracks().catch(() => null),
      ]);
      setDrones(d.data);
      setStats(s.data);
      if (t) {
        setPolledTracks(t.data.tracks || []);
        setPolledSwarm(Boolean(t.data.swarm_detected));
      }
    } catch (e) {
      console.error("DroneWatch load error:", e);
    } finally {
      setLoading(false);
    }
  }, []);
  useEffect(() => {
    load();
    const t = setInterval(load, 10000);
    return () => clearInterval(t);
  }, [load]);
  useEffect(() => {
    if (!latestDetection) return;
    const now = Date.now();
    const incoming = (latestDetection.detections || []).filter(
      (d) => d.type === "drone",
    );
    setLiveMap((prev) => {
      const next = {};
      Object.entries(prev).forEach(([k, v]) => {
        if (now - v.seen < 5000) next[k] = v;
      });
      incoming.forEach((d) => {
        next[`${latestDetection.camera_id}:${d.track_id}`] = {
          ...d,
          camera_id: latestDetection.camera_id,
          swarm: Boolean(latestDetection.swarm),
          seen: now,
        };
      });
      return next;
    });
  }, [latestDetection]);
  useEffect(() => {
    const t = setInterval(() => {
      const now = Date.now();
      setLiveMap((prev) => {
        const entries = Object.entries(prev).filter(
          ([, v]) => now - v.seen < 5000,
        );
        return entries.length === Object.keys(prev).length
          ? prev
          : Object.fromEntries(entries);
      });
    }, 2000);
    return () => clearInterval(t);
  }, []);
  const liveTracks = useMemo(() => {
    const live = Object.values(liveMap);
    return connected && live.length > 0 ? live : polledTracks;
  }, [liveMap, polledTracks, connected]);
  const swarmActive =
    polledSwarm || Object.values(liveMap).some((t) => t.swarm);
  useEffect(() => {
    if (!latestDrone) return;
    setLiveDetections((prev) => [
      {
        ...latestDrone,
        _liveKey: Date.now() + Math.random(),
      },
      ...prev.slice(0, 11),
    ]);
  }, [latestDrone]);
  const riskColor = {
    critical: "var(--sev-critical)",
    warning: "var(--sev-high)",
    caution: "var(--sev-medium)",
    clear: "var(--sev-low)",
  };
  return (
    <div
      style={{
        padding: 28,
      }}
    >
      {}
      <h1
        style={{
          fontFamily: "var(--font-display)",
          fontSize: 20,
          color: "var(--qf-cyan)",
          letterSpacing: 3,
          marginBottom: 4,
        }}
      >
        DRONE WATCH
      </h1>
      <div
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 11,
          color: "var(--qf-text-muted)",
          marginBottom: 24,
        }}
      >
        UAV / AERIAL INTRUSION DETECTION SYSTEM
      </div>

      {swarmActive && (
        <div
          className="qf-card"
          style={{
            marginBottom: 20,
            borderColor: "var(--qf-red)",
            color: "var(--qf-red)",
            fontFamily: "var(--font-mono)",
            fontSize: 12,
            letterSpacing: 2,
          }}
        >
          COORDINATED MULTI-DRONE ACTIVITY DETECTED
        </div>
      )}

      {}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(4, 1fr)",
          gap: 14,
          marginBottom: 24,
        }}
      >
        {[
          ["TOTAL DETECTED", stats?.total_detections ?? "-", "var(--qf-cyan)"],
          ["LAST 24H", stats?.last_24h ?? "-", "var(--qf-yellow)"],
          ["UNAUTHORIZED", stats?.unauthorized ?? "-", "var(--qf-red)"],
          [
            "THREAT %",
            stats ? `${stats.threat_percentage}%` : "-",
            "var(--qf-orange)",
          ],
        ].map(([l, v, c]) => (
          <div
            key={l}
            className="qf-card"
            style={{
              textAlign: "center",
            }}
          >
            <div
              style={{
                fontFamily: "var(--font-display)",
                fontSize: 28,
                color: c,
                marginBottom: 6,
              }}
            >
              {v}
            </div>
            <div
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 9,
                color: "var(--qf-text-muted)",
                letterSpacing: 1,
              }}
            >
              {l}
            </div>
          </div>
        ))}
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "320px 1fr",
          gap: 16,
        }}
      >
        {}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 16,
          }}
        >
          <div className="qf-card">
            <div
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 10,
                color: "var(--qf-text-muted)",
                letterSpacing: 2,
                marginBottom: 14,
                textAlign: "center",
              }}
            >
              AERIAL RADAR DISPLAY
            </div>
            <DroneRadar drones={drones} tracks={liveTracks} />
            <div
              style={{
                textAlign: "center",
                marginTop: 14,
              }}
            >
              <span
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: 12,
                  color:
                    liveTracks.length > 0 || drones.length > 0
                      ? "var(--qf-red)"
                      : "var(--qf-green)",
                }}
              >
                {liveTracks.length > 0
                  ? `${liveTracks.length} UAV(s) TRACKED LIVE`
                  : drones.length > 0
                    ? `${drones.length} UAV(s) IN LOG`
                    : "AIRSPACE CLEAR"}
              </span>
            </div>
          </div>

          <div className="qf-card">
            <div
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 10,
                color: "var(--qf-text-muted)",
                letterSpacing: 2,
                marginBottom: 12,
              }}
            >
              LIVE TRACKS
            </div>
            {liveTracks.length === 0 ? (
              <div
                style={{
                  color: "var(--qf-text-muted)",
                  fontFamily: "var(--font-mono)",
                  fontSize: 11,
                }}
              >
                No active tracks
              </div>
            ) : (
              <table className="qf-table">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>CAM</th>
                    <th>TYPE</th>
                    <th>ALT</th>
                    <th>SPEED</th>
                    <th>THREAT</th>
                  </tr>
                </thead>
                <tbody>
                  {liveTracks.map((t) => (
                    <tr key={`${t.camera_id}:${t.track_id}`}>
                      <td
                        style={{
                          fontFamily: "var(--font-mono)",
                          fontSize: 11,
                        }}
                      >
                        #{t.track_id}
                      </td>
                      <td
                        style={{
                          fontFamily: "var(--font-mono)",
                          fontSize: 11,
                        }}
                      >
                        {String(t.camera_id).padStart(2, "0")}
                      </td>
                      <td
                        style={{
                          fontFamily: "var(--font-mono)",
                          fontSize: 11,
                          textTransform: "uppercase",
                        }}
                      >
                        {t.drone_type || "unknown"}
                      </td>
                      <td
                        style={{
                          fontFamily: "var(--font-mono)",
                          fontSize: 11,
                        }}
                      >
                        {typeof t.altitude_m === "number"
                          ? `${t.altitude_m.toFixed(0)}m`
                          : "-"}
                      </td>
                      <td
                        style={{
                          fontFamily: "var(--font-mono)",
                          fontSize: 11,
                        }}
                      >
                        {typeof t.speed_ms === "number"
                          ? `${t.speed_ms.toFixed(1)} m/s`
                          : "-"}
                      </td>
                      <td
                        style={{
                          fontFamily: "var(--font-mono)",
                          fontSize: 11,
                          color: threatColor(t.threat_score ?? 0),
                        }}
                      >
                        {typeof t.threat_score === "number"
                          ? `${(t.threat_score * 100).toFixed(0)}%`
                          : "-"}
                        {t.is_authorized ? " AUTH" : ""}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>

          {}
          <div
            className="qf-card"
            style={{
              flex: 1,
            }}
          >
            <div
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 10,
                color: "var(--qf-text-muted)",
                letterSpacing: 2,
                marginBottom: 12,
              }}
            >
              LIVE DRONE ALERTS
            </div>
            {liveDetections.length === 0 ? (
              <div
                style={{
                  color: "var(--qf-text-muted)",
                  fontFamily: "var(--font-mono)",
                  fontSize: 11,
                }}
              >
                No live detections…
              </div>
            ) : (
              liveDetections.map((d) => (
                <div
                  key={d._liveKey}
                  style={{
                    display: "flex",
                    gap: 10,
                    alignItems: "center",
                    padding: "8px 0",
                    borderBottom: "1px solid var(--qf-border)",
                    animation: "slide-in-right 0.3s ease",
                  }}
                >
                  <span
                    style={{
                      fontSize: 18,
                      color: "var(--qf-red)",
                    }}
                  >
                    ◈
                  </span>
                  <div
                    style={{
                      flex: 1,
                    }}
                  >
                    <div
                      style={{
                        fontWeight: 600,
                        fontSize: 12,
                        color: "var(--qf-red)",
                      }}
                    >
                      DRONE DETECTED
                    </div>
                    <div
                      style={{
                        fontFamily: "var(--font-mono)",
                        fontSize: 10,
                        color: "var(--qf-text-muted)",
                      }}
                    >
                      CAM {d.camera_id} · {(d.confidence * 100).toFixed(0)}% ·{" "}
                      {d.threat_level?.toUpperCase()}
                    </div>
                    {d.summary && (
                      <div
                        style={{
                          fontSize: 10,
                          color: "var(--qf-text-secondary)",
                          marginTop: 2,
                          lineHeight: 1.3,
                        }}
                      >
                        AI: {d.summary.slice(0, 80)}
                      </div>
                    )}
                  </div>
                  <span
                    className="badge badge-high"
                    style={{
                      fontSize: 9,
                    }}
                  >
                    THREAT
                  </span>
                </div>
              ))
            )}
          </div>
        </div>

        {}
        <div
          className="qf-card"
          style={{
            padding: 0,
            overflow: "hidden",
            alignSelf: "start",
          }}
        >
          <div
            style={{
              padding: "14px 20px",
              fontFamily: "var(--font-mono)",
              fontSize: 10,
              color: "var(--qf-text-muted)",
              letterSpacing: 2,
              borderBottom: "1px solid var(--qf-border)",
            }}
          >
            DRONE DETECTION LOG - LAST 2H
          </div>

          {loading ? (
            <div
              style={{
                textAlign: "center",
                padding: 40,
                color: "var(--qf-text-muted)",
                fontFamily: "var(--font-mono)",
              }}
            >
              LOADING…
            </div>
          ) : (
            <table className="qf-table">
              <thead>
                <tr>
                  <th>TIME</th>
                  <th>CAMERA</th>
                  <th>CONFIDENCE</th>
                  <th>TYPE</th>
                  <th>ALTITUDE</th>
                  <th>SPEED</th>
                  <th>RISK</th>
                  <th>AI ANALYSIS</th>
                </tr>
              </thead>
              <tbody>
                {drones.length === 0 ? (
                  <tr>
                    <td
                      colSpan={8}
                      style={{
                        textAlign: "center",
                        padding: 40,
                        color: "var(--qf-text-muted)",
                        fontFamily: "var(--font-mono)",
                      }}
                    >
                      NO DRONE DETECTIONS IN LOG
                    </td>
                  </tr>
                ) : (
                  drones.map((d) => (
                    <tr
                      key={d.id}
                      style={{
                        animation: "fade-in-up 0.2s ease",
                      }}
                    >
                      <td
                        style={{
                          fontFamily: "var(--font-mono)",
                          fontSize: 11,
                          whiteSpace: "nowrap",
                        }}
                      >
                        {new Date(d.timestamp).toLocaleTimeString()}
                      </td>
                      <td
                        style={{
                          fontFamily: "var(--font-mono)",
                          fontSize: 11,
                        }}
                      >
                        CAM {String(d.camera_id).padStart(2, "0")}
                      </td>
                      <td>
                        <div
                          style={{
                            fontFamily: "var(--font-mono)",
                            fontSize: 12,
                            color: "var(--qf-yellow)",
                          }}
                        >
                          {(d.confidence * 100).toFixed(1)}%
                        </div>
                      </td>
                      <td
                        style={{
                          fontFamily: "var(--font-mono)",
                          fontSize: 11,
                          color: "var(--qf-text-secondary)",
                          textTransform: "uppercase",
                        }}
                      >
                        {d.drone_type || "UNKNOWN"}
                      </td>
                      <td
                        style={{
                          fontFamily: "var(--font-mono)",
                          fontSize: 11,
                        }}
                      >
                        {d.estimated_altitude_m
                          ? `${d.estimated_altitude_m}m`
                          : "-"}
                      </td>
                      <td
                        style={{
                          fontFamily: "var(--font-mono)",
                          fontSize: 11,
                        }}
                      >
                        {d.estimated_speed_ms
                          ? `${d.estimated_speed_ms.toFixed(1)} m/s`
                          : "-"}
                      </td>
                      <td>
                        <span
                          className={`badge badge-${d.risk_level === "critical" ? "critical" : d.risk_level === "threat" ? "critical" : d.risk_level === "warning" ? "high" : "medium"}`}
                          style={{
                            fontSize: 9,
                          }}
                        >
                          {d.risk_level?.toUpperCase()}
                        </span>
                      </td>
                      <td
                        style={{
                          maxWidth: 200,
                        }}
                      >
                        {d.ai_analysis ? (
                          <div
                            style={{
                              fontSize: 11,
                              color: "var(--qf-text-muted)",
                              lineHeight: 1.3,
                            }}
                          >
                            {d.ai_analysis.slice(0, 70)}
                            {d.ai_analysis.length > 70 ? "…" : ""}
                          </div>
                        ) : (
                          "-"
                        )}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}
