import { useEffect, useState } from "react";

const int = new Intl.NumberFormat("en-US");
const compact = new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 });

export const fmt = {
  int: (n: number) => int.format(Math.round(n)),
  compact: (n: number) => compact.format(n),
  ratio: (n: number | null | undefined, digits = 3) => (n == null ? "–" : n.toFixed(digits)),
  pct: (n: number, digits = 1) => `${(n * 100).toFixed(digits)}%`,
  minutes: (m: number) => (m < 60 ? `${m.toFixed(0)} min` : `${(m / 60).toFixed(1)} h`),
  // Log timestamps are UTC epoch seconds; show them in UTC so they match the raw lines.
  dateTime: (s: number) =>
    new Date(s * 1000).toISOString().slice(0, 16).replace("T", " ") + " UTC",
  date: (s: number) => new Date(s * 1000).toISOString().slice(0, 10),
  hour: (s: number) => new Date(s * 1000).toISOString().slice(5, 16).replace("T", " "),
};

export type Async<T> =
  | { status: "loading" }
  | { status: "error"; error: Error }
  | { status: "ready"; data: T };

/** Run an async loader whenever deps change, discarding results from stale calls. */
export function useAsync<T>(load: () => Promise<T>, deps: unknown[]): Async<T> {
  const [state, setState] = useState<Async<T>>({ status: "loading" });
  useEffect(() => {
    let live = true;
    setState({ status: "loading" });
    load().then(
      (data) => live && setState({ status: "ready", data }),
      (error: Error) => live && setState({ status: "error", error }),
    );
    return () => {
      live = false;
    };
  }, deps);
  return state;
}

export type Theme = "light" | "dark";

/** Resolved theme: explicit choice if set, otherwise the OS preference. */
export function useTheme(): [Theme, (t: Theme) => void] {
  const media = window.matchMedia("(prefers-color-scheme: dark)");
  const [choice, setChoice] = useState<Theme | null>(() => {
    try {
      return (localStorage.getItem("theme") as Theme | null) ?? null;
    } catch {
      return null;
    }
  });
  const [system, setSystem] = useState<Theme>(media.matches ? "dark" : "light");

  useEffect(() => {
    const onChange = (e: MediaQueryListEvent) => setSystem(e.matches ? "dark" : "light");
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, [media]);

  const theme = choice ?? system;
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  const set = (t: Theme) => {
    setChoice(t);
    try {
      localStorage.setItem("theme", t);
    } catch {
      /* storage unavailable: choice lasts for this visit */
    }
  };
  return [theme, set];
}

/** Chart colours per theme: validated categorical slots 1-2 (blue, orange). */
export const CHART = {
  light: { series1: "#2a78d6", series2: "#eb6834", grid: "#e4e3df", axis: "#6b6a66", cursor: "#c9c8c2" },
  dark: { series1: "#3987e5", series2: "#d95926", grid: "#2e2e2b", axis: "#a3a29b", cursor: "#4a4a46" },
} as const;
