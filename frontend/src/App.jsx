import React, { createContext, useContext, useState, useEffect } from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
} from "react-router-dom";
import Layout from "./components/Layout";
import Home from "./pages/Home";
import Dashboard from "./pages/Dashboard";
import Cameras from "./pages/Cameras";
import Alerts from "./pages/Alerts";
import DroneWatch from "./pages/DroneWatch";
import MapView from "./pages/MapView";
import Geofences from "./pages/Geofences";
import Analytics from "./pages/Analytics";
import Settings from "./pages/Settings";
import Profile from "./pages/Profile";
import Login from "./pages/Login";
import SignUp from "./pages/SignUp";
import NotFound from "./pages/NotFound";
import { AuthProvider, useAuth } from "./context/AuthContext";
import { WebSocketProvider } from "./context/WebSocketContext";

// Guards the authenticated app shell - bounces signed-out visitors to /login.
function ProtectedRoute({ children }) {
  const { isAuthenticated, loading } = useAuth();
  if (loading) return <SplashScreen />;
  if (!isAuthenticated) return <Navigate to="/login" replace />;
  return children;
}

// Keeps signed-in users off the auth forms - sends them straight to the
// dashboard instead of showing Login/Sign Up again.
function PublicOnlyRoute({ children }) {
  const { isAuthenticated, loading } = useAuth();
  if (loading) return <SplashScreen />;
  if (isAuthenticated) return <Navigate to="/dashboard" replace />;
  return children;
}

function SplashScreen() {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        height: "100vh",
        background: "var(--qf-bg-deep)",
        gap: 24,
      }}
    >
      <div style={{ position: "relative", width: 80, height: 80 }}>
        <div
          style={{
            position: "absolute",
            inset: 0,
            borderRadius: "50%",
            border: "2px solid transparent",
            borderTopColor: "var(--qf-cyan)",
            animation: "rotate-ring 1s linear infinite",
          }}
        />
        <div
          style={{
            position: "absolute",
            inset: 8,
            borderRadius: "50%",
            border: "2px solid transparent",
            borderTopColor: "var(--qf-green)",
            animation: "rotate-ring 1.5s linear infinite reverse",
          }}
        />
        <div
          style={{
            position: "absolute",
            inset: 16,
            borderRadius: "50%",
            background: "var(--qf-cyan-dim)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 20,
          }}
        >
          ◈
        </div>
      </div>
      <div
        style={{
          fontFamily: "var(--font-display)",
          fontSize: 18,
          color: "var(--qf-cyan)",
          letterSpacing: 4,
        }}
      >
        QUANTUMFENCE
      </div>
      <div
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 12,
          color: "var(--qf-text-muted)",
        }}
      >
        INITIALIZING SYSTEMS...
      </div>
    </div>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <WebSocketProvider>
        <Router>
          <div className="scan-overlay" />
          <Routes>
            {/* Public marketing homepage - always the entry point */}
            <Route path="/" element={<Home />} />

            {/* Auth forms - redirect away if already signed in */}
            <Route
              path="/login"
              element={
                <PublicOnlyRoute>
                  <Login />
                </PublicOnlyRoute>
              }
            />
            <Route
              path="/signup"
              element={
                <PublicOnlyRoute>
                  <SignUp />
                </PublicOnlyRoute>
              }
            />

            {/* Authenticated app shell */}
            <Route
              path="/*"
              element={
                <ProtectedRoute>
                  <Layout>
                    <Routes>
                      <Route path="/dashboard" element={<Dashboard />} />
                      <Route path="/cameras" element={<Cameras />} />
                      <Route path="/alerts" element={<Alerts />} />
                      <Route path="/drones" element={<DroneWatch />} />
                      <Route path="/map" element={<MapView />} />
                      <Route path="/geofences" element={<Geofences />} />
                      <Route path="/analytics" element={<Analytics />} />
                      <Route path="/settings" element={<Settings />} />
                      <Route path="/profile" element={<Profile />} />
                      <Route path="*" element={<NotFound />} />
                    </Routes>
                  </Layout>
                </ProtectedRoute>
              }
            />
          </Routes>
        </Router>
      </WebSocketProvider>
    </AuthProvider>
  );
}
