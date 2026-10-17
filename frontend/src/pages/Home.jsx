import React from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

const FEATURES = [
  {
    icon: "◉",
    title: "Camera Network",
    text: "Onboard unlimited IP, RTSP, USB, and HTTP-MJPEG cameras with live WebSocket status and PTZ control.",
  },
  {
    icon: "◈",
    title: "AI Detection (YOLOv8)",
    text: "Real-time person, vehicle, and drone detection with multi-object tracking and swarm recognition.",
  },
  {
    icon: "◆",
    title: "Claude AI Threat Analysis",
    text: "Natural-language threat summaries, 0.0-1.0 risk scoring, and recommended operator actions.",
  },
  {
    icon: "⊕",
    title: "Geospatial Intelligence",
    text: "Draw geofence zones, visualize camera FOV, and overlay live threats on a tactical map.",
  },
  {
    icon: "◈",
    title: "Drone Watch",
    text: "Animated radar display with trajectory, altitude, and speed estimation for every intrusion.",
  },
  {
    icon: "▣",
    title: "Analytics & Reporting",
    text: "Detection timelines, alert breakdowns, and camera performance across 24h / 7d / 30d windows.",
  },
];

const STATS = [
  { value: "24/7", label: "AUTONOMOUS MONITORING" },
  { value: "<1s", label: "ALERT LATENCY" },
  { value: "64", label: "MAX CAMERAS / SITE" },
  { value: "3", label: "ACCESS ROLES" },
];

function GlowOrbs() {
  return (
    <>
      <div
        style={{
          position: "absolute",
          width: 700,
          height: 700,
          borderRadius: "50%",
          top: -280,
          left: "50%",
          transform: "translateX(-50%)",
          background:
            "radial-gradient(circle, var(--qf-cyan-glow) 0%, transparent 70%)",
          pointerEvents: "none",
        }}
      />
      <div
        style={{
          position: "absolute",
          width: 500,
          height: 500,
          borderRadius: "50%",
          bottom: -200,
          right: -150,
          background: "radial-gradient(circle, #00ff8811 0%, transparent 70%)",
          pointerEvents: "none",
        }}
      />
    </>
  );
}

function NavBar() {
  const { isAuthenticated, user } = useAuth();
  return (
    <header
      style={{
        position: "sticky",
        top: 0,
        zIndex: 50,
        background: "#020408cc",
        backdropFilter: "blur(10px)",
        borderBottom: "1px solid var(--qf-border)",
      }}
    >
      <div
        className="qf-container"
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          height: 68,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div
            style={{
              width: 34,
              height: 34,
              borderRadius: 9,
              background:
                "linear-gradient(135deg, var(--qf-cyan-dim), var(--qf-bg-surface))",
              border: "1px solid var(--qf-cyan)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 16,
              boxShadow: "0 0 16px var(--qf-cyan-dim)",
            }}
          >
            ◈
          </div>
          <span
            style={{
              fontFamily: "var(--font-display)",
              fontSize: 15,
              fontWeight: 900,
              color: "var(--qf-cyan)",
              letterSpacing: 3,
            }}
          >
            QUANTUMFENCE
          </span>
        </div>

        <nav style={{ display: "flex", alignItems: "center", gap: 10 }}>
          {isAuthenticated ? (
            <Link
              to="/dashboard"
              className="qf-btn qf-btn-primary"
              style={{ fontSize: 12 }}
            >
              {user?.username ? `${user.username.toUpperCase()} · ` : ""}
              GO TO DASHBOARD →
            </Link>
          ) : (
            <>
              <Link
                to="/login"
                className="qf-btn qf-btn-outline"
                style={{ fontSize: 12 }}
              >
                SIGN IN
              </Link>
              <Link
                to="/signup"
                className="qf-btn qf-btn-primary"
                style={{ fontSize: 12 }}
              >
                GET STARTED
              </Link>
            </>
          )}
        </nav>
      </div>
    </header>
  );
}

export default function Home() {
  const { isAuthenticated } = useAuth();
  const navigate = useNavigate();

  return (
    <div style={{ background: "var(--qf-bg-deep)", minHeight: "100vh" }}>
      <NavBar />

      {/* ── Hero ─────────────────────────────────────────────────────────── */}
      <section
        className="qf-grid-bg"
        style={{ position: "relative", overflow: "hidden" }}
      >
        <GlowOrbs />
        <div
          className="qf-container animate-fade-up"
          style={{
            position: "relative",
            padding: "96px 24px 80px",
            textAlign: "center",
          }}
        >
          <div
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 8,
              padding: "6px 14px",
              borderRadius: 20,
              border: "1px solid var(--qf-border-glow)",
              background: "var(--qf-cyan-glow)",
              fontFamily: "var(--font-mono)",
              fontSize: 11,
              color: "var(--qf-cyan)",
              letterSpacing: 1.5,
              marginBottom: 28,
            }}
          >
            <span className="status-dot status-online" />
            QUANTUM-ACCELERATED PERIMETER DEFENSE AI
          </div>

          <h1
            style={{
              fontFamily: "var(--font-display)",
              fontWeight: 900,
              fontSize: "clamp(32px, 5.5vw, 58px)",
              lineHeight: 1.15,
              letterSpacing: 1,
              color: "var(--qf-text-primary)",
              maxWidth: 900,
              margin: "0 auto 22px",
            }}
          >
            See every threat before it{" "}
            <span
              style={{
                color: "var(--qf-cyan)",
                textShadow: "0 0 30px var(--qf-cyan)",
              }}
            >
              reaches the fence
            </span>
          </h1>

          <p
            style={{
              fontSize: 17,
              lineHeight: 1.7,
              color: "var(--qf-text-secondary)",
              maxWidth: 620,
              margin: "0 auto 36px",
            }}
          >
            Multi-camera perimeter security with real-time drone detection,
            geofencing, and Claude AI threat analysis - built for security teams
            who can't afford to miss a signal.
          </p>

          <div
            style={{
              display: "flex",
              gap: 14,
              justifyContent: "center",
              flexWrap: "wrap",
            }}
          >
            {isAuthenticated ? (
              <button
                className="qf-btn qf-btn-primary"
                style={{ fontSize: 14, padding: "13px 28px" }}
                onClick={() => navigate("/dashboard")}
              >
                GO TO DASHBOARD →
              </button>
            ) : (
              <>
                <button
                  className="qf-btn qf-btn-primary"
                  style={{ fontSize: 14, padding: "13px 28px" }}
                  onClick={() => navigate("/signup")}
                >
                  DEPLOY QUANTUMFENCE →
                </button>
                <button
                  className="qf-btn qf-btn-outline"
                  style={{ fontSize: 14, padding: "13px 28px" }}
                  onClick={() => navigate("/login")}
                >
                  OPERATOR SIGN IN
                </button>
              </>
            )}
          </div>

          {/* Stats strip */}
          <div
            className="qf-card"
            style={{
              marginTop: 64,
              display: "grid",
              gridTemplateColumns: "repeat(4, 1fr)",
              gap: 8,
              maxWidth: 780,
              marginLeft: "auto",
              marginRight: "auto",
              textAlign: "center",
            }}
          >
            {STATS.map((s) => (
              <div key={s.label}>
                <div
                  style={{
                    fontFamily: "var(--font-display)",
                    fontSize: 26,
                    fontWeight: 900,
                    color: "var(--qf-cyan)",
                  }}
                >
                  {s.value}
                </div>
                <div
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontSize: 9,
                    color: "var(--qf-text-muted)",
                    letterSpacing: 1,
                    marginTop: 4,
                  }}
                >
                  {s.label}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Feature grid ─────────────────────────────────────────────────── */}
      <section style={{ padding: "40px 24px 100px" }}>
        <div className="qf-container">
          <div style={{ textAlign: "center", marginBottom: 48 }}>
            <div
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 11,
                color: "var(--qf-text-muted)",
                letterSpacing: 2,
                marginBottom: 10,
              }}
            >
              SYSTEM CAPABILITIES
            </div>
            <h2
              style={{
                fontFamily: "var(--font-display)",
                fontSize: 28,
                color: "var(--qf-text-primary)",
                letterSpacing: 1,
              }}
            >
              One command center. Every layer of defense.
            </h2>
          </div>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
              gap: 18,
            }}
          >
            {FEATURES.map((f, i) => (
              <div
                key={f.title}
                className="qf-card animate-fade-up"
                style={{ animationDelay: `${i * 0.05}s`, opacity: 0 }}
              >
                <div
                  style={{
                    width: 44,
                    height: 44,
                    borderRadius: 12,
                    background: "var(--qf-bg-surface)",
                    border: "1px solid var(--qf-border)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: 20,
                    color: "var(--qf-cyan)",
                    marginBottom: 16,
                  }}
                >
                  {f.icon}
                </div>
                <h3
                  style={{
                    fontFamily: "var(--font-display)",
                    fontSize: 15,
                    color: "var(--qf-text-primary)",
                    letterSpacing: 0.5,
                    marginBottom: 8,
                  }}
                >
                  {f.title}
                </h3>
                <p
                  style={{
                    fontSize: 13.5,
                    lineHeight: 1.6,
                    color: "var(--qf-text-secondary)",
                  }}
                >
                  {f.text}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── CTA banner ───────────────────────────────────────────────────── */}
      {!isAuthenticated && (
        <section style={{ padding: "0 24px 100px" }}>
          <div
            className="qf-container qf-card qf-grid-bg"
            style={{
              textAlign: "center",
              padding: "56px 24px",
              border: "1px solid var(--qf-border-glow)",
              boxShadow: "0 0 60px var(--qf-cyan-glow)",
            }}
          >
            <h2
              style={{
                fontFamily: "var(--font-display)",
                fontSize: 24,
                color: "var(--qf-text-primary)",
                marginBottom: 12,
              }}
            >
              Ready to harden your perimeter?
            </h2>
            <p
              style={{
                color: "var(--qf-text-secondary)",
                marginBottom: 28,
                maxWidth: 480,
                marginLeft: "auto",
                marginRight: "auto",
              }}
            >
              Create an operator account and connect your first camera in
              minutes.
            </p>
            <div
              style={{
                display: "flex",
                gap: 14,
                justifyContent: "center",
                flexWrap: "wrap",
              }}
            >
              <Link
                to="/signup"
                className="qf-btn qf-btn-primary"
                style={{ fontSize: 13 }}
              >
                CREATE ACCOUNT
              </Link>
              <Link
                to="/login"
                className="qf-btn qf-btn-outline"
                style={{ fontSize: 13 }}
              >
                SIGN IN
              </Link>
            </div>
          </div>
        </section>
      )}

      {/* ── Footer ───────────────────────────────────────────────────────── */}
      <footer
        style={{
          borderTop: "1px solid var(--qf-border)",
          padding: "28px 24px",
        }}
      >
        <div
          className="qf-container"
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            flexWrap: "wrap",
            gap: 12,
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            color: "var(--qf-text-muted)",
            letterSpacing: 1,
          }}
        >
          <span>
            QUANTUMFENCE v1.0.0 · QUANTUM-ACCELERATED PERIMETER DEFENSE
          </span>
          <span>ALL ACCESS ATTEMPTS ARE LOGGED AND MONITORED</span>
        </div>
      </footer>
    </div>
  );
}
