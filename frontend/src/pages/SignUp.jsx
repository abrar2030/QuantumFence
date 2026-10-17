import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

const initialForm = {
  username: "",
  email: "",
  full_name: "",
  password: "",
  confirmPassword: "",
};

function validate(form) {
  if (form.username.trim().length < 3) {
    return "Username must be at least 3 characters";
  }
  if (!/^[a-zA-Z0-9_.-]+$/.test(form.username.trim())) {
    return "Username may only contain letters, numbers, dots, dashes, underscores";
  }
  if (!/^\S+@\S+\.\S+$/.test(form.email)) {
    return "Enter a valid email address";
  }
  if (form.password.length < 8) {
    return "Password must be at least 8 characters";
  }
  if (form.password !== form.confirmPassword) {
    return "Passwords do not match";
  }
  return "";
}

export default function SignUp() {
  const [form, setForm] = useState(initialForm);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const { register } = useAuth();
  const navigate = useNavigate();

  const set = (k, v) => setForm((f) => ({ ...f, [k]: v }));

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    const validationError = validate(form);
    if (validationError) {
      setError(validationError);
      return;
    }

    setLoading(true);
    try {
      await register({
        username: form.username.trim(),
        email: form.email.trim(),
        password: form.password,
        full_name: form.full_name.trim() || undefined,
      });
      navigate("/dashboard");
    } catch (err) {
      setError(
        err.response?.data?.detail ||
          "Could not create account - please try again",
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "var(--qf-bg-deep)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        position: "relative",
        overflow: "hidden",
        padding: "40px 0",
      }}
      className="qf-grid-bg"
    >
      {/* Background glow orbs */}
      <div
        style={{
          position: "absolute",
          width: 600,
          height: 600,
          borderRadius: "50%",
          top: -200,
          right: -200,
          background:
            "radial-gradient(circle, var(--qf-cyan-glow) 0%, transparent 70%)",
          pointerEvents: "none",
        }}
      />
      <div
        style={{
          position: "absolute",
          width: 400,
          height: 400,
          borderRadius: "50%",
          bottom: -100,
          left: -100,
          background: "radial-gradient(circle, #00ff8811 0%, transparent 70%)",
          pointerEvents: "none",
        }}
      />

      <div
        style={{
          width: "100%",
          maxWidth: 460,
          padding: 24,
          position: "relative",
        }}
      >
        {/* Logo */}
        <Link
          to="/"
          style={{
            textDecoration: "none",
            display: "block",
            textAlign: "center",
            marginBottom: 32,
          }}
        >
          <div
            style={{
              display: "inline-flex",
              alignItems: "center",
              justifyContent: "center",
              width: 64,
              height: 64,
              borderRadius: 18,
              background:
                "linear-gradient(135deg, var(--qf-cyan-dim), var(--qf-bg-surface))",
              border: "1px solid var(--qf-cyan)",
              fontSize: 28,
              marginBottom: 16,
              boxShadow: "0 0 40px var(--qf-cyan-dim)",
            }}
          >
            ◈
          </div>
          <div
            style={{
              fontFamily: "var(--font-display)",
              fontSize: 24,
              fontWeight: 900,
              color: "var(--qf-cyan)",
              letterSpacing: 5,
              textShadow: "0 0 30px var(--qf-cyan)",
            }}
          >
            QUANTUMFENCE
          </div>
        </Link>

        {/* Sign Up Card */}
        <div
          className="qf-card"
          style={{
            background: "var(--qf-bg-card)",
            border: "1px solid var(--qf-border)",
            boxShadow: "0 0 60px var(--qf-cyan-glow)",
          }}
        >
          <div style={{ textAlign: "center", marginBottom: 24 }}>
            <div
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 11,
                color: "var(--qf-text-muted)",
                letterSpacing: 2,
              }}
            >
              CREATE OPERATOR ACCOUNT
            </div>
          </div>

          <form onSubmit={handleSubmit}>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: 14,
                marginBottom: 16,
              }}
            >
              <div>
                <label className="field-label">USERNAME *</label>
                <input
                  className="qf-input"
                  type="text"
                  autoComplete="username"
                  placeholder="j.operator"
                  value={form.username}
                  onChange={(e) => set("username", e.target.value)}
                  required
                />
              </div>
              <div>
                <label className="field-label">FULL NAME</label>
                <input
                  className="qf-input"
                  type="text"
                  autoComplete="name"
                  placeholder="Jane Operator"
                  value={form.full_name}
                  onChange={(e) => set("full_name", e.target.value)}
                />
              </div>
            </div>

            <div style={{ marginBottom: 16 }}>
              <label className="field-label">EMAIL *</label>
              <input
                className="qf-input"
                type="email"
                autoComplete="email"
                placeholder="jane@yourcompany.com"
                value={form.email}
                onChange={(e) => set("email", e.target.value)}
                required
              />
            </div>

            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: 14,
                marginBottom: 8,
              }}
            >
              <div>
                <label className="field-label">PASSWORD *</label>
                <input
                  className="qf-input"
                  type="password"
                  autoComplete="new-password"
                  placeholder="Min. 8 characters"
                  value={form.password}
                  onChange={(e) => set("password", e.target.value)}
                  required
                />
              </div>
              <div>
                <label className="field-label">CONFIRM PASSWORD *</label>
                <input
                  className="qf-input"
                  type="password"
                  autoComplete="new-password"
                  placeholder="Repeat password"
                  value={form.confirmPassword}
                  onChange={(e) => set("confirmPassword", e.target.value)}
                  required
                />
              </div>
            </div>

            <div
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 10,
                color: "var(--qf-text-muted)",
                marginBottom: 20,
                letterSpacing: 0.5,
              }}
            >
              New accounts are created with OPERATOR access. An administrator
              can adjust roles later in Settings → Users.
            </div>

            {error && (
              <div
                style={{
                  background: "var(--qf-red-dim)",
                  border: "1px solid var(--qf-red)",
                  borderRadius: 8,
                  padding: "10px 14px",
                  marginBottom: 16,
                  fontFamily: "var(--font-mono)",
                  fontSize: 12,
                  color: "var(--qf-red)",
                }}
              >
                {error}
              </div>
            )}

            <button
              type="submit"
              className="qf-btn qf-btn-primary"
              style={{
                width: "100%",
                justifyContent: "center",
                fontSize: 13,
                padding: "13px",
              }}
              disabled={loading}
            >
              {loading ? (
                <span
                  style={{ display: "flex", alignItems: "center", gap: 10 }}
                >
                  <span
                    style={{
                      width: 14,
                      height: 14,
                      border: "2px solid transparent",
                      borderTopColor: "var(--qf-bg-deep)",
                      borderRadius: "50%",
                      animation: "rotate-ring 0.8s linear infinite",
                      display: "inline-block",
                    }}
                  />
                  CREATING ACCOUNT...
                </span>
              ) : (
                "CREATE ACCOUNT"
              )}
            </button>
          </form>

          <div
            style={{
              marginTop: 20,
              textAlign: "center",
              fontFamily: "var(--font-mono)",
              fontSize: 11,
              color: "var(--qf-text-muted)",
            }}
          >
            Already have access?{" "}
            <Link to="/login" className="qf-link">
              Sign in
            </Link>
          </div>
        </div>

        {/* Footer */}
        <div
          style={{
            textAlign: "center",
            marginTop: 24,
            fontFamily: "var(--font-mono)",
            fontSize: 10,
            color: "var(--qf-text-muted)",
            letterSpacing: 1,
            lineHeight: 1.8,
          }}
        >
          <Link to="/" className="qf-link" style={{ fontSize: 10 }}>
            ← Back to home
          </Link>
        </div>
      </div>
    </div>
  );
}
