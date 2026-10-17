import React, { useState } from "react";
import { authApi } from "../services/api";
import { useAuth } from "../context/AuthContext";

function Banner({ kind, children }) {
  if (!children) return null;
  const ok = kind === "success";
  return (
    <div
      style={{
        background: ok ? "var(--qf-green-dim)" : "var(--qf-red-dim)",
        border: `1px solid ${ok ? "var(--qf-green)" : "var(--qf-red)"}`,
        borderRadius: 8,
        padding: "10px 14px",
        marginBottom: 16,
        fontFamily: "var(--font-mono)",
        fontSize: 12,
        color: ok ? "var(--qf-green)" : "var(--qf-red)",
      }}
    >
      {ok ? "+" : "!"} {children}
    </div>
  );
}

const ROLE_COPY = {
  admin: "Full system access - manages users, cameras, and configuration.",
  operator: "Manages cameras, alerts, and geofences. Cannot manage users.",
  viewer: "Read-only access to dashboards, alerts, and analytics.",
};

export default function Profile() {
  const { user, updateUser } = useAuth();

  const [profileForm, setProfileForm] = useState({
    full_name: user?.full_name || "",
    email: user?.email || "",
  });
  const [profileSaving, setProfileSaving] = useState(false);
  const [profileMsg, setProfileMsg] = useState({ kind: "", text: "" });

  const [pwForm, setPwForm] = useState({
    current_password: "",
    new_password: "",
    confirm_password: "",
  });
  const [pwSaving, setPwSaving] = useState(false);
  const [pwMsg, setPwMsg] = useState({ kind: "", text: "" });

  const handleProfileSave = async (e) => {
    e.preventDefault();
    setProfileMsg({ kind: "", text: "" });
    setProfileSaving(true);
    try {
      const r = await authApi.updateMe({
        full_name: profileForm.full_name,
        email: profileForm.email,
      });
      updateUser(r.data);
      setProfileMsg({ kind: "success", text: "Profile updated" });
    } catch (err) {
      setProfileMsg({
        kind: "error",
        text: err.response?.data?.detail || "Could not update profile",
      });
    } finally {
      setProfileSaving(false);
    }
  };

  const handlePasswordChange = async (e) => {
    e.preventDefault();
    setPwMsg({ kind: "", text: "" });

    if (pwForm.new_password.length < 8) {
      setPwMsg({
        kind: "error",
        text: "New password must be at least 8 characters",
      });
      return;
    }
    if (pwForm.new_password !== pwForm.confirm_password) {
      setPwMsg({ kind: "error", text: "New passwords do not match" });
      return;
    }

    setPwSaving(true);
    try {
      await authApi.changePassword(
        pwForm.current_password,
        pwForm.new_password,
      );
      setPwMsg({ kind: "success", text: "Password changed" });
      setPwForm({
        current_password: "",
        new_password: "",
        confirm_password: "",
      });
    } catch (err) {
      setPwMsg({
        kind: "error",
        text: err.response?.data?.detail || "Could not change password",
      });
    } finally {
      setPwSaving(false);
    }
  };

  return (
    <div style={{ padding: 28, maxWidth: 740 }}>
      <h1
        style={{
          fontFamily: "var(--font-display)",
          fontSize: 20,
          color: "var(--qf-cyan)",
          letterSpacing: 3,
          marginBottom: 4,
        }}
      >
        ACCOUNT
      </h1>
      <div
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 11,
          color: "var(--qf-text-muted)",
          marginBottom: 24,
        }}
      >
        Manage your profile and credentials
      </div>

      {/* Identity summary */}
      <div
        className="qf-card"
        style={{
          display: "flex",
          alignItems: "center",
          gap: 18,
          marginBottom: 20,
        }}
      >
        <div
          style={{
            width: 56,
            height: 56,
            borderRadius: 14,
            flexShrink: 0,
            background: "var(--qf-bg-surface)",
            border: "1px solid var(--qf-cyan)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 22,
            color: "var(--qf-cyan)",
            fontWeight: 700,
            fontFamily: "var(--font-display)",
            boxShadow: "0 0 20px var(--qf-cyan-dim)",
          }}
        >
          {user?.username?.[0]?.toUpperCase() || "U"}
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: 16, fontWeight: 700 }}>
            {user?.full_name || user?.username}
          </div>
          <div
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: 11,
              color: "var(--qf-text-muted)",
              marginTop: 2,
            }}
          >
            @{user?.username}
          </div>
        </div>
        <span
          className={`badge ${
            user?.role === "admin"
              ? "badge-critical"
              : user?.role === "operator"
                ? "badge-medium"
                : "badge-low"
          }`}
        >
          {user?.role}
        </span>
      </div>

      <div
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 11,
          color: "var(--qf-text-secondary)",
          marginBottom: 24,
          lineHeight: 1.6,
        }}
      >
        {ROLE_COPY[user?.role] || ""}
      </div>

      {/* Profile form */}
      <div className="qf-card" style={{ marginBottom: 20 }}>
        <h3
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            color: "var(--qf-text-muted)",
            letterSpacing: 2,
            marginBottom: 18,
          }}
        >
          PROFILE INFORMATION
        </h3>

        <form onSubmit={handleProfileSave}>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: 14,
              marginBottom: 18,
            }}
          >
            <div>
              <label className="field-label">USERNAME</label>
              <input
                className="qf-input"
                value={user?.username || ""}
                disabled
                style={{ opacity: 0.6, cursor: "not-allowed" }}
              />
            </div>
            <div>
              <label className="field-label">FULL NAME</label>
              <input
                className="qf-input"
                value={profileForm.full_name}
                onChange={(e) =>
                  setProfileForm((f) => ({ ...f, full_name: e.target.value }))
                }
                placeholder="Your full name"
              />
            </div>
            <div style={{ gridColumn: "1/-1" }}>
              <label className="field-label">EMAIL</label>
              <input
                className="qf-input"
                type="email"
                value={profileForm.email}
                onChange={(e) =>
                  setProfileForm((f) => ({ ...f, email: e.target.value }))
                }
                placeholder="you@yourcompany.com"
                required
              />
            </div>
          </div>

          <Banner kind={profileMsg.kind}>{profileMsg.text}</Banner>

          <button
            type="submit"
            className="qf-btn qf-btn-primary"
            disabled={profileSaving}
          >
            {profileSaving ? "SAVING..." : "SAVE CHANGES"}
          </button>
        </form>
      </div>

      {/* Password form */}
      <div className="qf-card">
        <h3
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            color: "var(--qf-text-muted)",
            letterSpacing: 2,
            marginBottom: 18,
          }}
        >
          CHANGE PASSWORD
        </h3>

        <form onSubmit={handlePasswordChange}>
          <div style={{ display: "grid", gap: 14, marginBottom: 18 }}>
            <div>
              <label className="field-label">CURRENT PASSWORD</label>
              <input
                className="qf-input"
                type="password"
                autoComplete="current-password"
                value={pwForm.current_password}
                onChange={(e) =>
                  setPwForm((f) => ({ ...f, current_password: e.target.value }))
                }
                required
              />
            </div>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: 14,
              }}
            >
              <div>
                <label className="field-label">NEW PASSWORD</label>
                <input
                  className="qf-input"
                  type="password"
                  autoComplete="new-password"
                  value={pwForm.new_password}
                  onChange={(e) =>
                    setPwForm((f) => ({ ...f, new_password: e.target.value }))
                  }
                  required
                />
              </div>
              <div>
                <label className="field-label">CONFIRM NEW PASSWORD</label>
                <input
                  className="qf-input"
                  type="password"
                  autoComplete="new-password"
                  value={pwForm.confirm_password}
                  onChange={(e) =>
                    setPwForm((f) => ({
                      ...f,
                      confirm_password: e.target.value,
                    }))
                  }
                  required
                />
              </div>
            </div>
          </div>

          <Banner kind={pwMsg.kind}>{pwMsg.text}</Banner>

          <button
            type="submit"
            className="qf-btn qf-btn-outline"
            disabled={pwSaving}
          >
            {pwSaving ? "UPDATING..." : "UPDATE PASSWORD"}
          </button>
        </form>
      </div>
    </div>
  );
}
