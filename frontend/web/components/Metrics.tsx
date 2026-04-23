"use client";
import { useEffect, useState } from "react";
import axios from "axios";
import dynamic from "next/dynamic";
import { BASE_URL } from "@/lib/api";

// Dynamic import for Plotly to prevent SSR issues in Next.js
const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

export default function Metrics() {
  const [data, setData] = useState<any>(null);

  useEffect(() => {
    axios.get(`${BASE_URL}/metrics`).then((res) => setData(res.data));
  }, []);

  if (!data) return (
    <div className="flex items-center justify-center h-48">
      <div className="w-6 h-6 border-2 border-white/20 border-t-white rounded-full animate-spin" />
    </div>
  );

  const sharedLayout = {
    autosize: true,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#666", family: "Inter, sans-serif", size: 10 },
    margin: { l: 30, r: 10, b: 30, t: 30 },
    xaxis: { gridcolor: "#1a1a1a", zerolinecolor: "#333" },
    yaxis: { gridcolor: "#1a1a1a", zerolinecolor: "#333" },
  };

  return (
    <div className="space-y-12">
      {/* KPI Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="p-8 bg-white/[0.02] border border-white/[0.05] rounded-[2rem] flex flex-col items-center justify-center text-center">
          <span className="text-[10px] font-bold uppercase tracking-[0.2em] text-gray-500 mb-2">Top-1 Accuracy</span>
          <span className="text-5xl font-semibold tracking-tighter text-white">
            {(data.top1 * 100).toFixed(1)}%
          </span>
        </div>
        <div className="p-8 bg-white/[0.02] border border-white/[0.05] rounded-[2rem] flex flex-col items-center justify-center text-center">
          <span className="text-[10px] font-bold uppercase tracking-[0.2em] text-gray-500 mb-2">Top-5 Accuracy</span>
          <span className="text-5xl font-semibold tracking-tighter text-white">
            {(data.top5 * 100).toFixed(1)}%
          </span>
        </div>
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        
        {/* ROC Curve Box */}
        <div className="space-y-4">
          <h3 className="text-sm font-medium text-gray-400 px-2">
            Receiver Operating Characteristic <span className="text-blue-500 ml-2">AUC {data.roc.auc.toFixed(2)}</span>
          </h3>
          <div className="bg-white/[0.02] border border-white/[0.05] rounded-[2rem] p-4 overflow-hidden">
            <Plot
              className="w-full"
              data={[{
                x: data.roc.fpr,
                y: data.roc.tpr,
                type: "scatter",
                mode: "lines",
                line: { color: "#3b82f6", width: 3, shape: 'spline' },
                fill: 'tozeroy',
                fillcolor: 'rgba(59, 130, 246, 0.05)'
              }]}
              layout={{ ...sharedLayout, height: 240 }}
              config={{ displayModeBar: false }}
            />
          </div>
        </div>

        {/* PR Curve Box */}
        <div className="space-y-4">
          <h3 className="text-sm font-medium text-gray-400 px-2">Precision-Recall Analysis</h3>
          <div className="bg-white/[0.02] border border-white/[0.05] rounded-[2rem] p-4 overflow-hidden">
            <Plot
              className="w-full"
              data={[{
                x: data.pr.recall,
                y: data.pr.precision,
                type: "scatter",
                mode: "lines",
                line: { color: "#ffffff", width: 2, shape: 'spline' },
              }]}
              layout={{ ...sharedLayout, height: 240 }}
              config={{ displayModeBar: false }}
            />
          </div>
        </div>

        {/* Confusion Matrix Section */}
<div className="space-y-4">
  <h3 className="text-[10px] font-bold uppercase tracking-[0.3em] text-gray-600 px-2">
    Confusion Matrix Analysis
  </h3>
  
  <div className="bg-white/[0.02] border border-white/[0.05] rounded-[2.5rem] p-8">
    <div className="grid grid-cols-2 gap-4 max-w-sm mx-auto">
      
      {/* True Negative */}
      <div className="aspect-square flex flex-col items-center justify-center rounded-3xl bg-white/5 border border-white/5 group hover:bg-white/[0.08] transition-colors">
        <span className="text-[10px] font-bold text-gray-500 tracking-widest mb-1">TN</span>
        <span className="text-3xl font-semibold">{data.confusion[0][0]}</span>
      </div>

      {/* False Positive */}
      <div className="aspect-square flex flex-col items-center justify-center rounded-3xl bg-red-500/10 border border-red-500/20 group hover:bg-red-500/20 transition-colors">
        <span className="text-[10px] font-bold text-red-400 tracking-widest mb-1">FP</span>
        <span className="text-3xl font-semibold text-red-500">{data.confusion[0][1]}</span>
      </div>

      {/* False Negative */}
      <div className="aspect-square flex flex-col items-center justify-center rounded-3xl bg-orange-500/10 border border-orange-500/20 group hover:bg-orange-500/20 transition-colors">
        <span className="text-[10px] font-bold text-orange-400 tracking-widest mb-1">FN</span>
        <span className="text-3xl font-semibold text-orange-500">{data.confusion[1][0]}</span>
      </div>

      {/* True Positive */}
      <div className="aspect-square flex flex-col items-center justify-center rounded-3xl bg-green-500/10 border border-green-500/20 group hover:bg-green-500/20 transition-colors">
        <span className="text-[10px] font-bold text-green-400 tracking-widest mb-1">TP</span>
        <span className="text-3xl font-semibold text-green-500">{data.confusion[1][1]}</span>
      </div>

    </div>
    
    {/* Sub-labeling for Matrix Context */}
    <div className="mt-6 flex justify-around text-[10px] font-medium text-gray-600 uppercase tracking-tighter">
      <span>Predicted Negative</span>
      <span>Predicted Positive</span>
    </div>
  </div>
</div>

      </div>
    </div>
  );
}
