import React, {
  createContext,
  useContext,
  useEffect,
  useRef,
  useState,
  useCallback,
} from "react";
import { useAuth } from "./AuthContext";
const WsContext = createContext(null);
const MAX_BACKOFF_MS = 30000;
export function WebSocketProvider({ children }) {
  const { isAuthenticated } = useAuth();
  const wsRef = useRef(null);
  const reconnectTimer = useRef(null);
  const attempts = useRef(0);
  const closedByClient = useRef(false);
  const listeners = useRef({});
  const [connected, setConnected] = useState(false);
  const [latestAlert, setLatestAlert] = useState(null);
  const [latestDetection, setLatestDetection] = useState(null);
  const [latestDrone, setLatestDrone] = useState(null);
  const [cameraStatuses, setCameraStatuses] = useState({});
  const emit = useCallback((event, data) => {
    (listeners.current[event] || []).forEach((cb) => cb(data));
  }, []);
  const connect = useCallback(() => {
    const token = localStorage.getItem("qf_token");
    if (!token || closedByClient.current) return;
    const base =
      import.meta.env.VITE_WS_URL ||
      (import.meta.env.DEV
        ? "ws://localhost:8000"
        : `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}`);
    const clientId = `web_${Date.now()}_${Math.random().toString(36).slice(2)}`;
    const url = `${base}/ws/${clientId}?token=${encodeURIComponent(token)}`;
    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;
      ws.onopen = () => {
        attempts.current = 0;
        setConnected(true);
        clearTimeout(reconnectTimer.current);
        ws._pingInterval = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(
              JSON.stringify({
                type: "ping",
              }),
            );
          }
        }, 25000);
      };
      ws.onmessage = ({ data: raw }) => {
        let msg;
        try {
          msg = JSON.parse(raw);
        } catch {
          return;
        }
        const { type, data, camera_id, status } = msg;
        switch (type) {
          case "alert":
            setLatestAlert(data);
            emit("alert", data);
            break;
          case "detection":
            setLatestDetection(data);
            emit("detection", data);
            break;
          case "drone_detection":
            setLatestDrone(data);
            emit("drone_detection", data);
            break;
          case "camera_status":
            setCameraStatuses((prev) => ({
              ...prev,
              [String(camera_id)]: status,
            }));
            emit("camera_status", {
              camera_id,
              status,
            });
            break;
          case "camera_added":
            emit("camera_added", msg);
            break;
          case "pong":
          case "connected":
            break;
          default:
            emit(type, msg);
        }
      };
      ws.onclose = () => {
        setConnected(false);
        clearInterval(ws._pingInterval);
        if (closedByClient.current) return;
        const delay = Math.min(MAX_BACKOFF_MS, 1000 * 2 ** attempts.current);
        attempts.current += 1;
        reconnectTimer.current = setTimeout(connect, delay);
      };
      ws.onerror = () => ws.close();
    } catch {
      const delay = Math.min(MAX_BACKOFF_MS, 1000 * 2 ** attempts.current);
      attempts.current += 1;
      reconnectTimer.current = setTimeout(connect, delay);
    }
  }, [emit]);
  useEffect(() => {
    if (!isAuthenticated) return undefined;
    closedByClient.current = false;
    attempts.current = 0;
    connect();
    return () => {
      closedByClient.current = true;
      clearTimeout(reconnectTimer.current);
      clearInterval(wsRef.current?._pingInterval);
      wsRef.current?.close();
      setConnected(false);
    };
  }, [connect, isAuthenticated]);
  const on = useCallback((event, callback) => {
    if (!listeners.current[event]) listeners.current[event] = [];
    listeners.current[event].push(callback);
    return () => {
      listeners.current[event] = (listeners.current[event] || []).filter(
        (cb) => cb !== callback,
      );
    };
  }, []);
  const send = useCallback((msg) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(msg));
    }
  }, []);
  const startCamera = useCallback(
    (cameraId) => {
      send({
        type: "start_camera",
        camera_id: cameraId,
      });
    },
    [send],
  );
  return (
    <WsContext.Provider
      value={{
        connected,
        latestAlert,
        latestDetection,
        latestDrone,
        cameraStatuses,
        on,
        send,
        startCamera,
      }}
    >
      {children}
    </WsContext.Provider>
  );
}
export const useWebSocket = () => useContext(WsContext);
