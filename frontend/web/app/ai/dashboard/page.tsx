"use client";

import { useEffect, useState } from "react";
import axios from "axios";
import dynamic from "next/dynamic";
import Link from "next/link";
import { BASE_URL } from "@/lib/api";

// Dynamic imports for performance
const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });
import ScatterPlot from "@/components/ScatterPlot";
import Controls from "@/components/Controls";
import Metrics from "@/components/Metrics";
import Upload from "@/components/Upload";

export default function UnifiedDashboard() {
  // --- Workspace State ---
  const [projectionData, setProjectionData] = useState([]);
  const [method, setMethod] = useState("umap");
  const [model, setModel] = useState("clip");
  const [isSyncing, setIsSyncing] = useState(false);

  // --- Analytics State ---
  const [analytics, setAnalytics] = useState<any>({
    metrics: {},
    curves: {},
    perClass: [],
    outliers: [],
    fusion: [],
    missed: [],
  });
  const [loadingAnalytics, setLoadingAnalytics] = useState(true);

  // 1. Fetch Projection (Workspace)
  const fetchProjection = async () => {
    setIsSyncing(true);
    try {
      const res = await axios.get(`${BASE_URL}/projection?method=${method}&model=${model}`);
      setProjectionData(res.data);
    } catch (err) {
      console.error("Projection Error:", err);
    } finally {
      setIsSyncing(false);
    }
  };

  // 2. Fetch Performance Logs (Analytics)
  const fetchAnalytics = async () => {
    try {
      const [m, c, pc, out, fus, mis] = await Promise.all([
        axios.get(`${BASE_URL}/metrics`),
        axios.get(`${BASE_URL}/curves`),
        axios.get(`${BASE_URL}/per-class`),
        axios.get(`${BASE_URL}/outliers`),
        axios.get(`${BASE_URL}/fusion`),
        axios.get(`${BASE_URL}/missed`),
      ]);
      setAnalytics({
        metrics: m.data,
        curves: c.data,
        perClass: pc.data,
        outliers: out.data,
        fusion: fus.data.results,
        missed: mis.data,
      });
    } catch (err) {
      console.error("Analytics Sync Error:", err);
    } finally {
      setLoadingAnalytics(false);
    }
  };

  useEffect(() => {
    fetchProjection();
    fetchAnalytics();
  }, [method, model]);

  const chartLayout = {
    autosize: true,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#666", family: "Inter", size: 10 },
    margin: { l: 30, r: 10, b: 30, t: 30 },
    xaxis: { gridcolor: "#1a1a1a", zeroline: false },
    yaxis: { gridcolor: "#1a1a1a", zeroline: false },
  };

  return (
    <div className="min-h-screen bg-[#050505] text-white selection:bg-blue-500/30">
      
      {/* --- PREMIUM NAVIGATION --- */}
      <nav className="h-20 border-b border-white/[0.05] flex items-center justify-between px-10 sticky top-0 bg-[#050505]/80 backdrop-blur-xl z-50">
        <div className="flex items-center gap-4">
          <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center font-black italic text-white">L</div>
          <span className="text-sm font-bold tracking-[0.2em] uppercase">Latent.Explorer</span>
        </div>
        <div className="flex gap-8">
          <Link href="/ai" className="text-[10px] font-bold uppercase tracking-widest text-gray-400 hover:text-purple-400 transition-all">AI Insights 🤖</Link>
          <Link href="/advanced" className="text-[10px] font-bold uppercase tracking-widest text-gray-400 hover:text-blue-400 transition-all">Advanced 🔬</Link>
        </div>
      </nav>

      <main className="max-w-[1400px] mx-auto px-6 py-16 space-y-32">
        
        {/* --- SECTION 1: WORKSPACE & CONTROLS --- */}
        <section className="space-y-12">
          <header className="text-center space-y-4">
            <h1 className="text-6xl font-semibold tracking-tighter bg-gradient-to-b from-white to-white/40 bg-clip-text text-transparent">
              Workspace
            </h1>
            <div className="relative inline-block pt-4">
              {isSyncing && (
                <div className="absolute -top-4 left-1/2 -translate-x-1/2 flex items-center gap-2">
                  <div className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-ping" />
                  <span className="text-[8px] font-bold text-blue-500 uppercase tracking-[0.3em]">Syncing Latent Space</span>
                </div>
              )}
              <Controls method={method} setMethod={setMethod} model={model} setModel={setModel} />
            </div>
          </header>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Projection Chart */}
            <div className="lg:col-span-2 bg-[#0a0a0a] border border-white/[0.08] rounded-[3rem] p-4 shadow-2xl">
              <div className="aspect-video w-full rounded-[2.5rem] overflow-hidden bg-black">
                <ScatterPlot data={projectionData} />
              </div>
            </div>
            {/* Quick KPIs */}
            <div className="grid grid-cols-1 gap-4">
              <KPICard label="Top-1 Accuracy" value={analytics.metrics.top1} isPercent accentColor="text-blue-500" />
              <KPICard label="Top-5 Accuracy" value={analytics.metrics.top5} isPercent />
              <KPICard label="F1 Score" value={analytics.metrics.f1} />
            </div>
          </div>
        </section>

        {/* --- SECTION 2: SYSTEM VISUALS --- */}
        <section className="space-y-10">
          <div className="flex items-center gap-4">
            <h2 className="text-xs font-bold uppercase tracking-[0.3em] text-gray-600">Model Stability Curves</h2>
            <div className="h-[1px] flex-1 bg-white/[0.05]" />
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="bg-[#0a0a0a] border border-white/[0.08] rounded-[2.5rem] p-8">
               <h3 className="text-[10px] font-bold text-gray-500 uppercase tracking-widest mb-4 text-center">ROC Area</h3>
               <Plot
                 className="w-full"
                 data={[{ x: analytics.curves.fpr, y: analytics.curves.tpr, type: "scatter", mode: "lines", line: { color: "#3b82f6", width: 3, shape: 'spline' }, fill: 'tozeroy', fillcolor: 'rgba(59,130,246,0.05)' }]}
                 layout={{ ...chartLayout, height: 280 }}
                 config={{ displayModeBar: false }}
               />
            </div>
            <div className="bg-[#0a0a0a] border border-white/[0.08] rounded-[2.5rem] p-8">
               <h3 className="text-[10px] font-bold text-gray-500 uppercase tracking-widest mb-4 text-center">Precision Recall</h3>
               <Plot
                 className="w-full"
                 data={[{ x: analytics.curves.recall, y: analytics.curves.precision, type: "scatter", mode: "lines", line: { color: "#ffffff", width: 2, shape: 'spline' } }]}
                 layout={{ ...chartLayout, height: 280 }}
                 config={{ displayModeBar: false }}
               />
            </div>
          </div>
        </section>

        {/* --- SECTION 3: CLASS ANALYSIS --- */}
        <section className="space-y-10">
          <div className="flex items-center gap-4">
            <h2 className="text-xs font-bold uppercase tracking-[0.3em] text-gray-600">Metric Evaluation</h2>
            <div className="h-[1px] flex-1 bg-white/[0.05]" />
          </div>
          <div className="bg-[#0a0a0a] border border-white/[0.08] rounded-[3rem] p-12">
            <Metrics />
          </div>
        </section>

        {/* --- SECTION 4: DIAGNOSTICS (OUTLIERS & MISSES) --- */}
        <section className="grid grid-cols-1 lg:grid-cols-2 gap-12">
          <div className="space-y-6">
            <h2 className="text-xs font-bold text-red-500 uppercase tracking-[0.4em] text-center">Inference Failure Log</h2>
            <div className="space-y-4">
              {analytics.missed.slice(0, 4).map((m: any, i: number) => (
                <div key={i} className="bg-red-500/5 border border-red-500/10 rounded-3xl p-6 flex justify-between items-center group hover:bg-red-500/10 transition-all">
                  <div>
                    <span className="text-[10px] font-bold text-red-400 uppercase block mb-1">{m.class_name}</span>
                    <p className="text-sm text-gray-400 italic">"{m.caption_short}"</p>
                  </div>
                  <div className="text-right">
                    <span className="text-xl font-mono font-light text-red-500">{m.score.toFixed(2)}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="space-y-6">
            <h2 className="text-xs font-bold text-blue-500 uppercase tracking-[0.4em] text-center">Anomaly Detection</h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {analytics.outliers.slice(0, 4).map((o: any, i: number) => (
                <div key={i} className={`p-6 rounded-3xl border transition-all ${o.is_outlier ? 'bg-orange-500/5 border-orange-500/20' : 'bg-white/[0.02] border-white/[0.05]'}`}>
                  <p className="text-xs text-gray-500 line-clamp-2 mb-4">"{o.caption}"</p>
                  <div className="flex justify-between items-center">
                    <span className={`text-[9px] font-bold uppercase tracking-widest ${o.is_outlier ? 'text-orange-400' : 'text-green-500'}`}>
                      {o.is_outlier ? 'Outlier' : 'Stable'}
                    </span>
                    <span className="text-[10px] font-mono text-gray-700">{o.score.toFixed(2)}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* --- SECTION 5: INFERENCE SIMULATION --- */}
        <section className="space-y-10 pb-20">
          <div className="flex items-center gap-4">
            <h2 className="text-xs font-bold uppercase tracking-[0.3em] text-gray-600">Simulate Inference</h2>
            <div className="h-[1px] flex-1 bg-white/[0.05]" />
          </div>
          <div className="bg-[#0a0a0a] border border-white/[0.08] rounded-[3rem] p-12">
            <Upload />
          </div>
        </section>

      </main>

      {/* Global CSS */}
      <style jsx global>{`
        ::-webkit-scrollbar { width: 5px; }
        ::-webkit-scrollbar-track { background: #050505; }
        ::-webkit-scrollbar-thumb { background: #1a1a1a; border-radius: 10px; }
        html { scroll-behavior: smooth; }
      `}</style>
    </div>
  );
}

function KPICard({ label, value, isPercent, accentColor = "text-white" }: any) {
  const formatted = value !== undefined 
    ? (isPercent ? `${(value * 100).toFixed(1)}%` : value.toFixed(3)) 
    : "0.000";

  return (
    <div className="bg-[#0a0a0a] border border-white/[0.08] rounded-[2.5rem] p-8 flex flex-col items-center justify-center text-center group hover:border-white/20 transition-all shadow-xl">
      <span className="text-[9px] font-bold uppercase tracking-[0.3em] text-gray-500 mb-2">{label}</span>
      <span className={`text-4xl font-semibold tracking-tighter ${accentColor}`}>
        {formatted}
      </span>
    </div>
  );
}
