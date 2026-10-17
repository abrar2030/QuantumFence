import React, { useState, useEffect, useCallback } from "react";
import { geofenceApi } from "../services/api";

const FENCE_TYPES = ["polygon", "circle"];

function emptyPoint() {
  return { lng: "", lat: "" };
}

// ─── Add/Edit Modal ──────────────────────────────────────────────────────────
function GeofenceModal({ geofence, onClose, onSave }) {
  const isPolygon = (geofence?.fence_type || "polygon") === "polygon";

  const [form, setForm] = useState(
    geofence
      ? {
          ...geofence,
          points: isPolygon
            ? (geofence.coordinates || []).map(([lng, lat]) => ({
                lng: String(lng),
                lat: String(lat),
              }))
            : [emptyPoint(), emptyPoint(), emptyPoint()],
        }
      : {
          name: "",
          description: "",
          fence_type: "polygon",
          color: "#FF4444",
          buffer_meters: 10,
          alert_on_entry: true,
          alert_on_exit: false,
          center_lat: "",
          center_lng: "",
          radius_meters: 100,
          points: [emptyPoint(), emptyPoint(), emptyPoint()],
        },
  );
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  const set = (k, v) => setForm((f) => ({ ...f, [k]: v }));

  const setPoint = (idx, key, value) =>
    setForm((f) => ({
      ...f,
      points: f.points.map((p, i) => (i === idx ? { ...p, [key]: value } : p)),
    }));

  const addPoint = () =>
    setForm((f) => ({ ...f, points: [...f.points, emptyPoint()] }));

  const removePoint = (idx) =>
    setForm((f) => ({
      ...f,
      points: f.points.filter((_, i) => i !== idx),
    }));

  const handleSave = async () => {
    if (!form.name.trim()) {
      setError("Zone name is required");
      return;
    }

    let payload = {
      name: form.name.trim(),
      description: form.description || "",
      fence_type: form.fence_type,
      buffer_meters: Number(form.buffer_meters) || 10,
      alert_on_entry: !!form.alert_on_entry,
      alert_on_exit: !!form.alert_on_exit,
      color: form.color || "#FF4444",
    };

    if (form.fence_type === "circle") {
      if (form.center_lat === "" || form.center_lng === "") {
        setError(
          "Center latitude and longitude are required for a circle zone",
        );
        return;
      }
      payload = {
        ...payload,
        coordinates: [],
        center_lat: Number(form.center_lat),
        center_lng: Number(form.center_lng),
        radius_meters: Number(form.radius_meters) || 100,
      };
    } else {
      const pts = form.points.filter((p) => p.lng !== "" && p.lat !== "");
      if (pts.length < 3) {
        setError("A polygon zone needs at least 3 points");
        return;
      }
      payload = {
        ...payload,
        coordinates: pts.map((p) => [Number(p.lng), Number(p.lat)]),
        center_lat: null,
        center_lng: null,
        radius_meters: null,
      };
    }

    setSaving(true);
    setError("");
    try {
      await onSave(geofence ? { ...payload, id: geofence.id } : payload);
      onClose();
    } catch (e) {
      setError(e.response?.data?.detail || "Error saving geofence");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        background: "#000000cc",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 1000,
        padding: 20,
      }}
    >
      <div
        className="qf-card"
        style={{
          width: "100%",
          maxWidth: 640,
          maxHeight: "90vh",
          overflowY: "auto",
          animation: "fade-in-up 0.25s ease",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            marginBottom: 24,
          }}
        >
          <h2
            style={{
              fontFamily: "var(--font-display)",
              fontSize: 16,
              color: "var(--qf-cyan)",
              letterSpacing: 2,
            }}
          >
            {geofence ? "EDIT GEOFENCE ZONE" : "NEW GEOFENCE ZONE"}
          </h2>
          <button
            onClick={onClose}
            style={{
              background: "none",
              border: "none",
              color: "var(--qf-text-muted)",
              cursor: "pointer",
              fontSize: 20,
            }}
          >
            ×
          </button>
        </div>

        <div style={{ display: "grid", gap: 14 }}>
          <div
            style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}
          >
            <div style={{ gridColumn: "1/-1" }}>
              <label className="field-label">ZONE NAME *</label>
              <input
                className="qf-input"
                value={form.name}
                onChange={(e) => set("name", e.target.value)}
                placeholder="e.g. North Perimeter"
              />
            </div>

            <div style={{ gridColumn: "1/-1" }}>
              <label className="field-label">DESCRIPTION</label>
              <input
                className="qf-input"
                value={form.description || ""}
                onChange={(e) => set("description", e.target.value)}
                placeholder="Optional notes about this zone"
              />
            </div>

            <div>
              <label className="field-label">FENCE TYPE</label>
              <select
                className="qf-select"
                style={{ width: "100%" }}
                value={form.fence_type}
                disabled={!!geofence}
                onChange={(e) => set("fence_type", e.target.value)}
              >
                {FENCE_TYPES.map((t) => (
                  <option key={t} value={t}>
                    {t.toUpperCase()}
                  </option>
                ))}
              </select>
              {geofence && (
                <div
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontSize: 9,
                    color: "var(--qf-text-muted)",
                    marginTop: 4,
                  }}
                >
                  Type can't change after creation - delete and recreate
                  instead.
                </div>
              )}
            </div>

            <div>
              <label className="field-label">ZONE COLOR</label>
              <input
                type="color"
                value={form.color}
                onChange={(e) => set("color", e.target.value)}
                style={{
                  width: "100%",
                  height: 40,
                  border: "1px solid var(--qf-border)",
                  borderRadius: 6,
                  background: "var(--qf-bg-surface)",
                  cursor: "pointer",
                }}
              />
            </div>

            <div>
              <label className="field-label">BUFFER (METERS)</label>
              <input
                className="qf-input"
                type="number"
                min="0"
                value={form.buffer_meters}
                onChange={(e) => set("buffer_meters", e.target.value)}
              />
            </div>

            <div style={{ display: "flex", alignItems: "flex-end", gap: 16 }}>
              <label className="qf-checkbox-row">
                <input
                  type="checkbox"
                  checked={!!form.alert_on_entry}
                  onChange={(e) => set("alert_on_entry", e.target.checked)}
                />
                ALERT ON ENTRY
              </label>
              <label className="qf-checkbox-row">
                <input
                  type="checkbox"
                  checked={!!form.alert_on_exit}
                  onChange={(e) => set("alert_on_exit", e.target.checked)}
                />
                ALERT ON EXIT
              </label>
            </div>
          </div>

          <hr
            style={{ border: "none", borderTop: "1px solid var(--qf-border)" }}
          />

          {form.fence_type === "circle" ? (
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr 1fr",
                gap: 14,
              }}
            >
              <div>
                <label className="field-label">CENTER LATITUDE</label>
                <input
                  className="qf-input"
                  type="number"
                  step="any"
                  value={form.center_lat}
                  onChange={(e) => set("center_lat", e.target.value)}
                  placeholder="33.6844"
                />
              </div>
              <div>
                <label className="field-label">CENTER LONGITUDE</label>
                <input
                  className="qf-input"
                  type="number"
                  step="any"
                  value={form.center_lng}
                  onChange={(e) => set("center_lng", e.target.value)}
                  placeholder="73.0479"
                />
              </div>
              <div>
                <label className="field-label">RADIUS (METERS)</label>
                <input
                  className="qf-input"
                  type="number"
                  min="1"
                  value={form.radius_meters}
                  onChange={(e) => set("radius_meters", e.target.value)}
                />
              </div>
            </div>
          ) : (
            <div>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  marginBottom: 10,
                }}
              >
                <label className="field-label" style={{ marginBottom: 0 }}>
                  BOUNDARY POINTS (MIN. 3)
                </label>
                <button
                  type="button"
                  onClick={addPoint}
                  className="qf-btn qf-btn-outline"
                  style={{ padding: "5px 12px", fontSize: 11 }}
                >
                  + ADD POINT
                </button>
              </div>

              <div style={{ display: "grid", gap: 8 }}>
                {form.points.map((p, idx) => (
                  <div
                    key={idx}
                    style={{
                      display: "grid",
                      gridTemplateColumns: "24px 1fr 1fr 32px",
                      gap: 8,
                      alignItems: "center",
                    }}
                  >
                    <div
                      style={{
                        fontFamily: "var(--font-mono)",
                        fontSize: 10,
                        color: "var(--qf-text-muted)",
                        textAlign: "center",
                      }}
                    >
                      {idx + 1}
                    </div>
                    <input
                      className="qf-input"
                      type="number"
                      step="any"
                      placeholder="Latitude"
                      value={p.lat}
                      onChange={(e) => setPoint(idx, "lat", e.target.value)}
                    />
                    <input
                      className="qf-input"
                      type="number"
                      step="any"
                      placeholder="Longitude"
                      value={p.lng}
                      onChange={(e) => setPoint(idx, "lng", e.target.value)}
                    />
                    <button
                      type="button"
                      onClick={() => removePoint(idx)}
                      disabled={form.points.length <= 3}
                      title={
                        form.points.length <= 3
                          ? "A polygon needs at least 3 points"
                          : "Remove point"
                      }
                      style={{
                        background: "none",
                        border: "none",
                        color:
                          form.points.length <= 3
                            ? "var(--qf-text-muted)"
                            : "var(--qf-red)",
                        cursor:
                          form.points.length <= 3 ? "not-allowed" : "pointer",
                        fontSize: 16,
                        opacity: form.points.length <= 3 ? 0.4 : 1,
                      }}
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {error && (
            <div
              style={{
                background: "var(--qf-red-dim)",
                border: "1px solid var(--qf-red)",
                borderRadius: 8,
                padding: "10px 14px",
                fontFamily: "var(--font-mono)",
                fontSize: 12,
                color: "var(--qf-red)",
              }}
            >
              {error}
            </div>
          )}

          <div style={{ display: "flex", gap: 10, justifyContent: "flex-end" }}>
            <button
              className="qf-btn qf-btn-outline"
              onClick={onClose}
              type="button"
            >
              CANCEL
            </button>
            <button
              className="qf-btn qf-btn-primary"
              onClick={handleSave}
              disabled={saving}
              type="button"
            >
              {saving ? "SAVING..." : geofence ? "SAVE CHANGES" : "CREATE ZONE"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Zone Card ───────────────────────────────────────────────────────────────
function ZoneCard({ gf, onEdit, onDelete, onToggleActive }) {
  return (
    <div className="qf-card">
      <div
        style={{
          display: "flex",
          alignItems: "flex-start",
          justifyContent: "space-between",
          marginBottom: 12,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span
            style={{
              width: 14,
              height: 14,
              borderRadius: 4,
              background: gf.color,
              flexShrink: 0,
              boxShadow: `0 0 10px ${gf.color}88`,
            }}
          />
          <div>
            <div style={{ fontWeight: 700, fontSize: 15 }}>{gf.name}</div>
            <div
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 10,
                color: "var(--qf-text-muted)",
                marginTop: 2,
              }}
            >
              {gf.fence_type.toUpperCase()}
              {gf.fence_type === "circle" && gf.radius_meters
                ? ` · ${gf.radius_meters}m RADIUS`
                : ` · ${gf.coordinates?.length || 0} POINTS`}
            </div>
          </div>
        </div>
        <span
          className={`badge ${gf.is_active ? "badge-low" : "badge-medium"}`}
        >
          {gf.is_active ? "ACTIVE" : "INACTIVE"}
        </span>
      </div>

      {gf.description && (
        <div
          style={{
            fontSize: 12.5,
            color: "var(--qf-text-secondary)",
            marginBottom: 12,
            lineHeight: 1.5,
          }}
        >
          {gf.description}
        </div>
      )}

      <div
        style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 14 }}
      >
        {gf.alert_on_entry && (
          <span className="badge badge-high">ALERT ON ENTRY</span>
        )}
        {gf.alert_on_exit && (
          <span className="badge badge-medium">ALERT ON EXIT</span>
        )}
        <span className="badge badge-low">BUFFER {gf.buffer_meters}M</span>
      </div>

      <div style={{ display: "flex", gap: 8 }}>
        <button
          className="qf-btn qf-btn-outline"
          style={{ flex: 1, justifyContent: "center", fontSize: 11 }}
          onClick={() => onEdit(gf)}
        >
          EDIT
        </button>
        <button
          className="qf-btn qf-btn-outline"
          style={{ flex: 1, justifyContent: "center", fontSize: 11 }}
          onClick={() => onToggleActive(gf)}
        >
          {gf.is_active ? "DEACTIVATE" : "ACTIVATE"}
        </button>
        <button
          className="qf-btn qf-btn-danger"
          style={{ justifyContent: "center", fontSize: 11 }}
          onClick={() => onDelete(gf.id)}
        >
          DELETE
        </button>
      </div>
    </div>
  );
}

// ─── Page ────────────────────────────────────────────────────────────────────
export default function Geofences() {
  const [zones, setZones] = useState([]);
  const [loading, setLoading] = useState(true);
  const [modal, setModal] = useState(null); // null | 'add' | zone_obj
  const [filter, setFilter] = useState("");

  const load = useCallback(async () => {
    try {
      const r = await geofenceApi.list();
      setZones(r.data);
    } catch (e) {
      console.error("Geofence load error:", e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const filtered = zones.filter(
    (z) =>
      !filter ||
      z.name.toLowerCase().includes(filter.toLowerCase()) ||
      z.fence_type.toLowerCase().includes(filter.toLowerCase()),
  );

  const handleSave = async (form) => {
    if (form.id) {
      const { id, ...rest } = form;
      await geofenceApi.update(id, rest);
    } else {
      await geofenceApi.create(form);
    }
    await load();
  };

  const handleDelete = async (id) => {
    if (!confirm("Delete this geofence zone? This cannot be undone.")) return;
    await geofenceApi.delete(id);
    await load();
  };

  const handleToggleActive = async (gf) => {
    await geofenceApi.update(gf.id, { is_active: !gf.is_active });
    await load();
  };

  const active = zones.filter((z) => z.is_active).length;

  return (
    <div style={{ padding: 28 }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: 24,
          flexWrap: "wrap",
          gap: 14,
        }}
      >
        <div>
          <h1
            style={{
              fontFamily: "var(--font-display)",
              fontSize: 20,
              color: "var(--qf-cyan)",
              letterSpacing: 3,
            }}
          >
            GEOFENCE ZONES
          </h1>
          <div
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: 11,
              color: "var(--qf-text-muted)",
              marginTop: 4,
            }}
          >
            {zones.length} zones &nbsp;·&nbsp;
            <span style={{ color: "var(--qf-green)" }}>{active} active</span>
          </div>
        </div>
        <button
          className="qf-btn qf-btn-primary"
          onClick={() => setModal("add")}
        >
          + NEW ZONE
        </button>
      </div>

      <input
        className="qf-input"
        placeholder="Search zones by name or type..."
        value={filter}
        onChange={(e) => setFilter(e.target.value)}
        style={{ maxWidth: 340, marginBottom: 22 }}
      />

      {loading ? (
        <div
          style={{
            color: "var(--qf-text-muted)",
            fontFamily: "var(--font-mono)",
          }}
        >
          LOADING ZONES...
        </div>
      ) : filtered.length === 0 ? (
        <div
          className="qf-card"
          style={{
            textAlign: "center",
            padding: 40,
            color: "var(--qf-text-muted)",
          }}
        >
          <div style={{ fontSize: 32, marginBottom: 10 }}>⊕</div>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: 12 }}>
            {zones.length === 0
              ? "No geofence zones yet - create one to start monitoring boundaries."
              : "No zones match your search."}
          </div>
        </div>
      ) : (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(320px, 1fr))",
            gap: 16,
          }}
        >
          {filtered.map((gf) => (
            <ZoneCard
              key={gf.id}
              gf={gf}
              onEdit={setModal}
              onDelete={handleDelete}
              onToggleActive={handleToggleActive}
            />
          ))}
        </div>
      )}

      {modal && (
        <GeofenceModal
          geofence={modal === "add" ? null : modal}
          onClose={() => setModal(null)}
          onSave={handleSave}
        />
      )}
    </div>
  );
}
