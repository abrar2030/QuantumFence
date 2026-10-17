import React, { useState, useEffect, useCallback } from "react";
import { systemApi, authApi, modelApi } from "../services/api";
import { useAuth } from "../context/AuthContext";
function Field({ label, defaultValue, type = "text", placeholder }) {
  return (
    <div>
      <label
        style={{
          display: "block",
          fontFamily: "var(--font-mono)",
          fontSize: 10,
          color: "var(--qf-text-muted)",
          marginBottom: 6,
          letterSpacing: 1,
        }}
      >
        {label.toUpperCase()}
      </label>
      <input
        className="qf-input"
        type={type}
        defaultValue={defaultValue}
        placeholder={placeholder || defaultValue || ""}
      />
    </div>
  );
}
function Toggle({ label, defaultOn = false }) {
  const [on, setOn] = useState(defaultOn);
  return (
    <button
      onClick={() => setOn((v) => !v)}
      style={{
        padding: "8px 14px",
        background: on ? "var(--qf-cyan-dim)" : "var(--qf-bg-surface)",
        border: `1px solid ${on ? "var(--qf-cyan)" : "var(--qf-border)"}`,
        borderRadius: 6,
        cursor: "pointer",
        fontFamily: "var(--font-mono)",
        fontSize: 10,
        color: on ? "var(--qf-cyan)" : "var(--qf-text-muted)",
        letterSpacing: 1,
      }}
    >
      {on ? "■" : "□"} {label}
    </button>
  );
}
const NEW_USER_INITIAL = {
  username: "",
  full_name: "",
  email: "",
  password: "",
  role: "operator",
};
function ModelRuntimePanel({ isAdmin }) {
  const [status, setStatus] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const load = useCallback(async () => {
    try {
      const r = await modelApi.status();
      setStatus(r.data);
      setError("");
    } catch (e) {
      setStatus(null);
      setError(e.response?.data?.detail || "Model runtime unavailable");
    }
  }, []);
  useEffect(() => {
    load();
    const t = setInterval(load, 10000);
    return () => clearInterval(t);
  }, [load]);
  const reload = async () => {
    setBusy(true);
    try {
      await modelApi.reload();
      await load();
    } catch (e) {
      setError(e.response?.data?.detail || "Reload failed");
    } finally {
      setBusy(false);
    }
  };
  const inference = status?.metrics?.timings?.inference;
  const rows = status
    ? [
        ["HEALTH", status.health?.toUpperCase()],
        ["DEVICE", String(status.device).toUpperCase()],
        ["DRONE MODE", String(status.drone_mode).toUpperCase()],
        ["GENERAL WEIGHTS", status.general_model?.weights || "-"],
        ["DRONE WEIGHTS", status.drone_model?.weights || "-"],
        [
          "INFERENCE",
          inference
            ? `${inference.avg_ms.toFixed(1)} ms avg / ${inference.p95_ms.toFixed(1)} ms p95 / ${inference.rate_per_s.toFixed(1)} fps`
            : "-",
        ],
      ]
    : [];
  return (
    <div
      className="qf-card"
      style={{
        display: "grid",
        gap: 10,
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <h3
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            color: "var(--qf-text-muted)",
            letterSpacing: 2,
          }}
        >
          MODEL RUNTIME
        </h3>
        {isAdmin && (
          <button
            className="qf-btn qf-btn-outline"
            onClick={reload}
            disabled={busy}
          >
            {busy ? "RELOADING" : "RELOAD MODELS"}
          </button>
        )}
      </div>
      {error && (
        <div
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            color: "var(--qf-red)",
          }}
        >
          {error}
        </div>
      )}
      {rows.map(([k, v]) => (
        <div
          key={k}
          style={{
            display: "flex",
            justifyContent: "space-between",
            gap: 16,
            fontFamily: "var(--font-mono)",
            fontSize: 11,
          }}
        >
          <span
            style={{
              color: "var(--qf-text-muted)",
            }}
          >
            {k}
          </span>
          <span
            style={{
              color: "var(--qf-text-primary)",
              textAlign: "right",
              wordBreak: "break-all",
            }}
          >
            {v}
          </span>
        </div>
      ))}
      {status?.errors?.map((e) => (
        <div
          key={e}
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            color: "var(--qf-orange)",
          }}
        >
          {e}
        </div>
      ))}
    </div>
  );
}
export default function Settings() {
  const [tab, setTab] = useState("system");
  const [saved, setSaved] = useState(false);
  const [health, setHealth] = useState(null);
  const { user } = useAuth();
  const isAdmin = user?.role === "admin";
  const [users, setUsers] = useState([]);
  const [usersLoading, setUsersLoading] = useState(false);
  const [newUser, setNewUser] = useState(NEW_USER_INITIAL);
  const [creatingUser, setCreatingUser] = useState(false);
  const [userMsg, setUserMsg] = useState({
    kind: "",
    text: "",
  });
  const [updatingUserId, setUpdatingUserId] = useState(null);
  const loadUsers = useCallback(async () => {
    if (!isAdmin) return;
    setUsersLoading(true);
    try {
      const r = await authApi.listUsers();
      setUsers(r.data);
    } catch (e) {
      setUserMsg({
        kind: "error",
        text: e.response?.data?.detail || "Could not load users",
      });
    } finally {
      setUsersLoading(false);
    }
  }, [isAdmin]);
  useEffect(() => {
    if (tab === "users") loadUsers();
  }, [tab, loadUsers]);
  const handleCreateUser = async (e) => {
    e.preventDefault();
    setUserMsg({
      kind: "",
      text: "",
    });
    if (
      !newUser.username.trim() ||
      !newUser.email.trim() ||
      !newUser.password
    ) {
      setUserMsg({
        kind: "error",
        text: "Username, email, and password are required",
      });
      return;
    }
    if (newUser.password.length < 8) {
      setUserMsg({
        kind: "error",
        text: "Password must be at least 8 characters",
      });
      return;
    }
    setCreatingUser(true);
    try {
      await authApi.register({
        username: newUser.username.trim(),
        email: newUser.email.trim(),
        password: newUser.password,
        full_name: newUser.full_name.trim() || undefined,
        role: newUser.role,
      });
      setUserMsg({
        kind: "success",
        text: `User "${newUser.username}" created`,
      });
      setNewUser(NEW_USER_INITIAL);
      await loadUsers();
    } catch (e) {
      setUserMsg({
        kind: "error",
        text: e.response?.data?.detail || "Could not create user",
      });
    } finally {
      setCreatingUser(false);
    }
  };
  const handleRoleChange = async (targetUser, role) => {
    setUpdatingUserId(targetUser.id);
    try {
      await authApi.updateUser(targetUser.id, {
        role,
      });
      await loadUsers();
    } catch (e) {
      setUserMsg({
        kind: "error",
        text: e.response?.data?.detail || "Could not update role",
      });
    } finally {
      setUpdatingUserId(null);
    }
  };
  const handleToggleUserActive = async (targetUser) => {
    setUpdatingUserId(targetUser.id);
    try {
      await authApi.updateUser(targetUser.id, {
        is_active: !targetUser.is_active,
      });
      await loadUsers();
    } catch (e) {
      setUserMsg({
        kind: "error",
        text: e.response?.data?.detail || "Could not update user",
      });
    } finally {
      setUpdatingUserId(null);
    }
  };
  const save = () => {
    setSaved(true);
    setTimeout(() => setSaved(false), 2500);
  };
  const checkHealth = async () => {
    try {
      const r = await systemApi.health();
      setHealth(r.data);
    } catch {
      setHealth({
        status: "error",
        message: "Backend not reachable",
      });
    }
  };
  const TABS = [
    {
      id: "system",
      label: "SYSTEM",
    },
    {
      id: "ai",
      label: "AI ENGINE",
    },
    {
      id: "notifications",
      label: "NOTIFICATIONS",
    },
    ...(isAdmin
      ? [
          {
            id: "users",
            label: "USERS",
          },
        ]
      : []),
    {
      id: "diagnostics",
      label: "DIAGNOSTICS",
    },
  ];
  return (
    <div
      style={{
        padding: 28,
      }}
    >
      <h1
        style={{
          fontFamily: "var(--font-display)",
          fontSize: 20,
          color: "var(--qf-cyan)",
          letterSpacing: 3,
          marginBottom: 24,
        }}
      >
        SYSTEM SETTINGS
      </h1>

      {}
      <div
        style={{
          display: "flex",
          gap: 0,
          marginBottom: 24,
          borderBottom: "1px solid var(--qf-border)",
        }}
      >
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            style={{
              padding: "10px 20px",
              background: "none",
              border: "none",
              cursor: "pointer",
              fontFamily: "var(--font-mono)",
              fontSize: 11,
              letterSpacing: 1.5,
              color: tab === t.id ? "var(--qf-cyan)" : "var(--qf-text-muted)",
              borderBottom: `2px solid ${tab === t.id ? "var(--qf-cyan)" : "transparent"}`,
              marginBottom: -1,
              transition: "all 0.2s",
            }}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div
        className="qf-card"
        style={{
          maxWidth: 740,
        }}
      >
        {}
        {tab === "system" && (
          <div
            style={{
              display: "grid",
              gap: 18,
            }}
          >
            <h3
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 11,
                color: "var(--qf-text-muted)",
                letterSpacing: 2,
              }}
            >
              CORE SYSTEM
            </h3>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: 14,
              }}
            >
              <Field label="Server Host" defaultValue="0.0.0.0" />
              <Field label="Server Port" defaultValue="8000" type="number" />
              <Field label="Worker Processes" defaultValue="4" type="number" />
              <Field label="Max Cameras" defaultValue="64" type="number" />
              <Field label="Frame Skip Rate" defaultValue="3" type="number" />
              <Field
                label="Detection Confidence"
                defaultValue="0.65"
                type="number"
              />
              <Field
                label="Alert Cooldown (sec)"
                defaultValue="30"
                type="number"
              />
              <Field
                label="Snapshot Retention (days)"
                defaultValue="30"
                type="number"
              />
            </div>
            <div>
              <label
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: 10,
                  color: "var(--qf-text-muted)",
                  letterSpacing: 1,
                  display: "block",
                  marginBottom: 8,
                }}
              >
                ALLOWED ORIGINS (comma-separated)
              </label>
              <input
                className="qf-input"
                defaultValue="http://localhost:3000,http://localhost:5173"
              />
            </div>
          </div>
        )}

        {}
        {tab === "ai" && (
          <div
            style={{
              display: "grid",
              gap: 18,
            }}
          >
            <h3
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 11,
                color: "var(--qf-text-muted)",
                letterSpacing: 2,
              }}
            >
              AI ENGINE CONFIGURATION
            </h3>
            <ModelRuntimePanel isAdmin={isAdmin} />
            <Field
              label="Anthropic API Key"
              type="password"
              placeholder="sk-ant-…"
            />
            <Field
              label="Claude Model"
              defaultValue="claude-sonnet-4-20250514"
            />
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: 14,
              }}
            >
              <Field
                label="Google Maps API Key"
                type="password"
                placeholder="AIza…"
              />
              <Field
                label="Google Earth Project"
                placeholder="your-gee-project-id"
              />
              <Field
                label="AI Confidence Threshold"
                defaultValue="0.65"
                type="number"
              />
              <Field
                label="Detection IOU Threshold"
                defaultValue="0.45"
                type="number"
              />
            </div>
            <div>
              <label
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: 10,
                  color: "var(--qf-text-muted)",
                  letterSpacing: 1,
                  display: "block",
                  marginBottom: 10,
                }}
              >
                FEATURES
              </label>
              <div
                style={{
                  display: "flex",
                  gap: 10,
                  flexWrap: "wrap",
                }}
              >
                <Toggle label="AI THREAT ANALYSIS" defaultOn={true} />
                <Toggle label="DRONE CLASSIFICATION" defaultOn={true} />
                <Toggle label="MULTI-THREAT DETECT" defaultOn={true} />
                <Toggle label="LOITERING DETECTION" defaultOn={true} />
                <Toggle label="SWARM DETECTION" defaultOn={true} />
              </div>
            </div>
          </div>
        )}

        {}
        {tab === "notifications" && (
          <div
            style={{
              display: "grid",
              gap: 18,
            }}
          >
            <h3
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 11,
                color: "var(--qf-text-muted)",
                letterSpacing: 2,
              }}
            >
              NOTIFICATION SETTINGS
            </h3>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: 14,
              }}
            >
              <Field label="SMTP Host" defaultValue="smtp.gmail.com" />
              <Field label="SMTP Port" defaultValue="587" type="number" />
              <Field
                label="SMTP Username"
                type="email"
                placeholder="alerts@yourdomain.com"
              />
              <Field label="SMTP Password" type="password" />
            </div>
            <Field
              label="Alert Recipients (comma-separated)"
              placeholder="sec@domain.com,ops@domain.com"
            />
            <Field
              label="Webhook URL"
              type="url"
              placeholder="https://hooks.slack.com/…"
            />
            <div>
              <label
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: 10,
                  color: "var(--qf-text-muted)",
                  letterSpacing: 1,
                  display: "block",
                  marginBottom: 10,
                }}
              >
                ALERT ON SEVERITY
              </label>
              <div
                style={{
                  display: "flex",
                  gap: 10,
                }}
              >
                <Toggle label="CRITICAL" defaultOn={true} />
                <Toggle label="HIGH" defaultOn={true} />
                <Toggle label="MEDIUM" defaultOn={false} />
                <Toggle label="LOW" defaultOn={false} />
              </div>
            </div>
          </div>
        )}

        {}
        {tab === "users" && isAdmin && (
          <div
            style={{
              display: "grid",
              gap: 18,
            }}
          >
            <h3
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 11,
                color: "var(--qf-text-muted)",
                letterSpacing: 2,
              }}
            >
              CREATE NEW USER
            </h3>

            <form onSubmit={handleCreateUser}>
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr",
                  gap: 14,
                  marginBottom: 14,
                }}
              >
                <div>
                  <label className="field-label">USERNAME *</label>
                  <input
                    className="qf-input"
                    value={newUser.username}
                    onChange={(e) =>
                      setNewUser((f) => ({
                        ...f,
                        username: e.target.value,
                      }))
                    }
                    placeholder="j.operator"
                  />
                </div>
                <div>
                  <label className="field-label">FULL NAME</label>
                  <input
                    className="qf-input"
                    value={newUser.full_name}
                    onChange={(e) =>
                      setNewUser((f) => ({
                        ...f,
                        full_name: e.target.value,
                      }))
                    }
                    placeholder="Jane Operator"
                  />
                </div>
                <div>
                  <label className="field-label">EMAIL *</label>
                  <input
                    className="qf-input"
                    type="email"
                    value={newUser.email}
                    onChange={(e) =>
                      setNewUser((f) => ({
                        ...f,
                        email: e.target.value,
                      }))
                    }
                    placeholder="jane@yourcompany.com"
                  />
                </div>
                <div>
                  <label className="field-label">PASSWORD *</label>
                  <input
                    className="qf-input"
                    type="password"
                    value={newUser.password}
                    onChange={(e) =>
                      setNewUser((f) => ({
                        ...f,
                        password: e.target.value,
                      }))
                    }
                    placeholder="Min. 8 characters"
                  />
                </div>
              </div>

              <div
                style={{
                  marginBottom: 14,
                }}
              >
                <label className="field-label">ROLE</label>
                <select
                  className="qf-select"
                  style={{
                    width: "100%",
                  }}
                  value={newUser.role}
                  onChange={(e) =>
                    setNewUser((f) => ({
                      ...f,
                      role: e.target.value,
                    }))
                  }
                >
                  {["admin", "operator", "viewer"].map((r) => (
                    <option key={r} value={r}>
                      {r.toUpperCase()}
                    </option>
                  ))}
                </select>
              </div>

              {userMsg.text && (
                <div
                  style={{
                    background:
                      userMsg.kind === "success"
                        ? "var(--qf-green-dim)"
                        : "var(--qf-red-dim)",
                    border: `1px solid ${userMsg.kind === "success" ? "var(--qf-green)" : "var(--qf-red)"}`,
                    borderRadius: 8,
                    padding: "10px 14px",
                    marginBottom: 14,
                    fontFamily: "var(--font-mono)",
                    fontSize: 12,
                    color:
                      userMsg.kind === "success"
                        ? "var(--qf-green)"
                        : "var(--qf-red)",
                  }}
                >
                  {userMsg.kind === "success" ? "+" : "!"} {userMsg.text}
                </div>
              )}

              <button
                type="submit"
                className="qf-btn qf-btn-primary"
                style={{
                  justifyContent: "center",
                }}
                disabled={creatingUser}
              >
                {creatingUser ? "CREATING..." : "+ CREATE USER"}
              </button>
            </form>

            <hr
              style={{
                border: "none",
                borderTop: "1px solid var(--qf-border)",
                margin: "4px 0",
              }}
            />

            <h3
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 11,
                color: "var(--qf-text-muted)",
                letterSpacing: 2,
              }}
            >
              ALL USERS {usersLoading ? "· LOADING…" : `· ${users.length}`}
            </h3>

            <div
              style={{
                overflowX: "auto",
              }}
            >
              <table className="qf-table">
                <thead>
                  <tr>
                    <th>USER</th>
                    <th>EMAIL</th>
                    <th>ROLE</th>
                    <th>STATUS</th>
                  </tr>
                </thead>
                <tbody>
                  {users.map((u) => {
                    const isSelf = u.id === user?.id;
                    const busy = updatingUserId === u.id;
                    return (
                      <tr key={u.id}>
                        <td>
                          <div
                            style={{
                              fontWeight: 600,
                            }}
                          >
                            {u.full_name || u.username}
                          </div>
                          <div
                            style={{
                              fontFamily: "var(--font-mono)",
                              fontSize: 10,
                              color: "var(--qf-text-muted)",
                            }}
                          >
                            @{u.username} {isSelf && "(you)"}
                          </div>
                        </td>
                        <td>{u.email}</td>
                        <td>
                          <select
                            className="qf-select"
                            value={u.role}
                            disabled={busy}
                            onChange={(e) =>
                              handleRoleChange(u, e.target.value)
                            }
                            style={{
                              fontSize: 11,
                              padding: "6px 10px",
                            }}
                          >
                            {["admin", "operator", "viewer"].map((r) => (
                              <option key={r} value={r}>
                                {r.toUpperCase()}
                              </option>
                            ))}
                          </select>
                        </td>
                        <td>
                          <button
                            onClick={() => handleToggleUserActive(u)}
                            disabled={busy || isSelf}
                            title={
                              isSelf
                                ? "You cannot deactivate your own account"
                                : undefined
                            }
                            className={`badge ${u.is_active ? "badge-low" : "badge-critical"}`}
                            style={{
                              border: "none",
                              cursor: isSelf ? "not-allowed" : "pointer",
                              opacity: isSelf ? 0.5 : 1,
                            }}
                          >
                            {u.is_active ? "ACTIVE" : "DISABLED"}
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                  {!usersLoading && users.length === 0 && (
                    <tr>
                      <td
                        colSpan={4}
                        style={{
                          textAlign: "center",
                          color: "var(--qf-text-muted)",
                        }}
                      >
                        No users found
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>

            <hr
              style={{
                border: "none",
                borderTop: "1px solid var(--qf-border)",
                margin: "4px 0",
              }}
            />
            <div
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 10,
                color: "var(--qf-text-muted)",
                letterSpacing: 1,
              }}
            >
              LOGGED IN AS:{" "}
              <span
                style={{
                  color: "var(--qf-cyan)",
                }}
              >
                {user?.username?.toUpperCase()}
              </span>{" "}
              &nbsp;·&nbsp; ROLE:{" "}
              <span
                style={{
                  color: "var(--qf-cyan)",
                }}
              >
                {user?.role?.toUpperCase()}
              </span>
            </div>
          </div>
        )}

        {}
        {tab === "diagnostics" && (
          <div
            style={{
              display: "grid",
              gap: 18,
            }}
          >
            <h3
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 11,
                color: "var(--qf-text-muted)",
                letterSpacing: 2,
              }}
            >
              SYSTEM DIAGNOSTICS
            </h3>
            <button
              className="qf-btn qf-btn-outline"
              style={{
                width: "fit-content",
                fontSize: 12,
              }}
              onClick={checkHealth}
            >
              ↻ CHECK BACKEND HEALTH
            </button>
            {health && (
              <div
                style={{
                  background: "var(--qf-bg-surface)",
                  borderRadius: 10,
                  padding: 16,
                  fontFamily: "var(--font-mono)",
                  fontSize: 12,
                }}
              >
                <div
                  style={{
                    marginBottom: 10,
                    color:
                      health.status === "healthy"
                        ? "var(--qf-green)"
                        : "var(--qf-red)",
                    fontWeight: 700,
                    fontSize: 14,
                  }}
                >
                  {health.status === "healthy"
                    ? "BACKEND HEALTHY"
                    : "BACKEND ERROR"}
                </div>
                {health.components &&
                  Object.entries(health.components).map(([k, v]) => (
                    <div
                      key={k}
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        marginBottom: 7,
                        borderBottom: "1px solid var(--qf-border)",
                        paddingBottom: 7,
                      }}
                    >
                      <span
                        style={{
                          color: "var(--qf-text-muted)",
                          letterSpacing: 1,
                        }}
                      >
                        {k.toUpperCase().replace(/_/g, " ")}
                      </span>
                      <span
                        style={{
                          color:
                            String(v).includes("load") ||
                            String(v).includes("connect") ||
                            String(v).includes("run") ||
                            typeof v === "number"
                              ? "var(--qf-green)"
                              : "var(--qf-yellow)",
                        }}
                      >
                        {String(v).toUpperCase()}
                      </span>
                    </div>
                  ))}
                <div
                  style={{
                    color: "var(--qf-text-muted)",
                    fontSize: 10,
                    marginTop: 4,
                  }}
                >
                  v{health.version}
                </div>
              </div>
            )}
            <div
              style={{
                display: "grid",
                gap: 10,
              }}
            >
              {[
                [
                  "API Docs",
                  `${import.meta.env.VITE_API_URL?.replace("/api", "") || "http://localhost:8000"}/api/docs`,
                ],
                [
                  "API ReDoc",
                  `${import.meta.env.VITE_API_URL?.replace("/api", "") || "http://localhost:8000"}/api/redoc`,
                ],
                ["Grafana", "http://localhost:3001"],
                ["Prometheus", "http://localhost:9090"],
              ].map(([l, url]) => (
                <div
                  key={l}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    background: "var(--qf-bg-surface)",
                    borderRadius: 8,
                    padding: "10px 14px",
                  }}
                >
                  <span
                    style={{
                      fontFamily: "var(--font-mono)",
                      fontSize: 11,
                      color: "var(--qf-text-muted)",
                    }}
                  >
                    {l}
                  </span>
                  <a
                    href={url}
                    target="_blank"
                    rel="noreferrer"
                    style={{
                      fontFamily: "var(--font-mono)",
                      fontSize: 11,
                      color: "var(--qf-cyan)",
                      textDecoration: "none",
                    }}
                  >
                    {url} ↗
                  </a>
                </div>
              ))}
            </div>
          </div>
        )}

        {}
        {tab !== "diagnostics" && tab !== "users" && (
          <div
            style={{
              marginTop: 24,
              display: "flex",
              justifyContent: "flex-end",
              alignItems: "center",
              gap: 14,
            }}
          >
            {saved && (
              <span
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: 12,
                  color: "var(--qf-green)",
                }}
              >
                SETTINGS SAVED
              </span>
            )}
            <button className="qf-btn qf-btn-primary" onClick={save}>
              SAVE SETTINGS
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
