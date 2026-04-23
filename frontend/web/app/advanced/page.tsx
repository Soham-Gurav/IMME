"use client";

import { useEffect, useState } from "react";
import axios from "axios";
import { BASE_URL } from "@/lib/api";

export default function AdvancedPage() {
  const [outliers, setOutliers] = useState<any[]>([]);
  const [fusion, setFusion] = useState<any[]>([]);

  useEffect(() => {
    axios.get(`${BASE_URL}/outliers`).then((res) => setOutliers(res.data));
    axios.get(`${BASE_URL}/fusion`).then((res) => setFusion(res.data.results));
  }, []);

  return (
    <div className="min-h-screen bg-[#050505] text-white selection:bg-blue-500/30">
      <main className="max-w-6xl mx-auto px-6 py-20 space-y-32">
        
        {/* Header Section */}
        <header className="space-y-4">
          <span className="text-[10px] font-bold tracking-[0.3em] text-blue-500 uppercase">
            Deep Diagnostic
          </span>
          <h1 className="text-5xl font-semibold tracking-tight">Advanced Analysis</h1>
          <p className="text-gray-500 max-w-xl text-lg">
            Detecting anomalies and cross-model fusion discrepancies within the latent space.
          </p>
        </header>

        {/* 🔴 OUTLIERS SECTION */}
        <section className="space-y-10">
          <div className="flex items-center gap-4">
            <h2 className="text-2xl font-medium tracking-tight">Outlier Detection</h2>
            <div className="h-[1px] flex-1 bg-white/[0.05]" />
            <span className="text-[10px] font-mono text-gray-600">LIMIT: 20 SAMPLES</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {outliers.slice(0, 20).map((o, i) => (
              <div 
                key={i} 
                className="group relative bg-[#0a0a0a] border border-white/[0.06] rounded-[2rem] p-6 hover:border-white/20 transition-all duration-500"
              >
                {/* Status Glow */}
                <div className={`absolute top-6 right-6 h-2 w-2 rounded-full blur-[2px] ${o.is_outlier ? "bg-red-500 shadow-[0_0_10px_rgba(239,68,68,0.5)]" : "bg-green-500 shadow-[0_0_10px_rgba(34,197,94,0.5)]"}`} />
                
                <div className="space-y-4">
                  <div className="text-[10px] font-bold text-gray-500 uppercase tracking-widest">
                    Sample {i + 1}
                  </div>
                  <p className="text-sm text-gray-300 leading-snug line-clamp-2 min-h-[2.5rem] group-hover:text-white transition-colors">
                    {o.caption}
                  </p>
                  <div className="pt-4 flex items-end justify-between border-t border-white/[0.03]">
                    <div>
                      <p className="text-[9px] text-gray-600 uppercase font-bold">Z-Score</p>
                      <p className="text-lg font-mono font-medium">{o.score.toFixed(3)}</p>
                    </div>
                    <span className={`text-[10px] font-bold px-2 py-1 rounded uppercase ${o.is_outlier ? "text-red-400 bg-red-400/10" : "text-green-400 bg-green-400/10"}`}>
                      {o.is_outlier ? "Anomalous" : "Stable"}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* 🟣 FUSION SECTION */}
        <section className="space-y-10 pb-32">
          <div className="flex items-center gap-4">
            <h2 className="text-2xl font-medium tracking-tight">Fusion Scoring</h2>
            <div className="h-[1px] flex-1 bg-white/[0.05]" />
            <span className="text-[10px] font-mono text-gray-600">MULTIMODAL DRIFT</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {fusion.slice(0, 20).map((f, i) => (
              <div 
                key={i} 
                className="bg-white/[0.02] border border-white/[0.05] rounded-3xl p-8 flex flex-col justify-between group hover:bg-white/[0.04] transition-all"
              >
                <p className="text-gray-400 text-sm font-medium mb-8 leading-relaxed italic group-hover:text-gray-200">
                  "{f.caption}"
                </p>

                <div className="space-y-6">
                  {/* Score Bar */}
                  <div className="space-y-2">
                    <div className="flex justify-between text-[10px] font-bold tracking-widest text-gray-500 uppercase">
                      <span>Fusion Metric</span>
                      <span>{f.fusion_score.toFixed(2)}</span>
                    </div>
                    <div className="h-1 w-full bg-white/5 rounded-full overflow-hidden">
                      <div 
                        className={`h-full transition-all duration-1000 ${f.fusion_pred ? "bg-red-500" : "bg-blue-500"}`}
                        style={{ width: `${Math.min(f.fusion_score * 100, 100)}%` }}
                      />
                    </div>
                  </div>

                  <div className="flex items-center justify-between">
                    <span className="text-[11px] font-medium text-gray-500">Evaluation</span>
                    <span className={`text-xs font-bold uppercase tracking-widest ${f.fusion_pred ? "text-red-500" : "text-blue-500"}`}>
                      {f.fusion_pred ? "Discordant" : "Aligned"}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>
      </main>
    </div>
  );
}
