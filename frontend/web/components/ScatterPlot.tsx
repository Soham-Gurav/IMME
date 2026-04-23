"use client";
import dynamic from "next/dynamic";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

export default function ScatterPlot({ data }: any) {
  if (!data || data.length === 0) return (
    <div className="flex flex-col items-center justify-center h-[500px] text-gray-600">
      <div className="w-12 h-12 border border-dashed border-gray-700 rounded-full animate-spin mb-4" />
      <p className="text-sm tracking-widest uppercase">Projecting Latent Space...</p>
    </div>
  );

  const x = data.map((p: any) => p.x);
  const y = data.map((p: any) => p.y);
  const text = data.map((p: any) => `<b>${p.image}</b><br>${p.caption}`);

  const labelMap: any = {};
  let labelIndex = 0;

  const colors = data.map((p: any) => {
    if (!(p.label in labelMap)) {
      labelMap[p.label] = labelIndex++;
    }
    return labelMap[p.label];
  });

  return (
    <div className="w-full h-full min-h-[600px] relative group">
      {/* Absolute UI labels for Axis context */}
      <div className="absolute bottom-4 left-6 z-10 pointer-events-none">
        <span className="text-[9px] font-bold tracking-[0.2em] text-gray-700 uppercase">Dimension A</span>
      </div>
      <div className="absolute top-1/2 left-2 z-10 pointer-events-none -rotate-90 origin-left">
        <span className="text-[9px] font-bold tracking-[0.2em] text-gray-700 uppercase">Dimension B</span>
      </div>

      <Plot
        className="w-full h-full"
        data={[
          {
            x,
            y,
            text,
            mode: "markers",
            type: "scatter",
            hoverinfo: "text",
            marker: {
              size: 7,
              color: colors,
              colorscale: [
                [0, '#3b82f6'],   // Blue
                [0.5, '#a855f7'], // Purple
                [1, '#ffffff']    // White
              ],
              showscale: true,
              colorbar: {
                thickness: 8,
                outlinewidth: 0,
                title: { text: "CLUSTER ID", font: { size: 10, color: '#444', family: 'Inter' } },
                tickfont: { color: '#444', size: 9 },
                len: 0.4,
                xpad: 30
              },
              line: {
                color: 'rgba(255,255,255,0.1)',
                width: 0.5
              },
              opacity: 0.8
            },
          }
        ]}
        layout={{
          autosize: true,
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          margin: { l: 40, r: 20, b: 40, t: 20 },
          dragmode: "pan",
          hovermode: "closest",
          font: { family: "Inter, sans-serif", color: "#444" },
          xaxis: {
            visible: true,
            showgrid: true,
            gridcolor: "rgba(255,255,255,0.03)", // Ultra light grid
            zeroline: true,
            zerolinecolor: "rgba(255,255,255,0.08)",
            tickfont: { size: 9, color: '#333' },
          },
          yaxis: {
            visible: true,
            showgrid: true,
            gridcolor: "rgba(255,255,255,0.03)",
            zeroline: true,
            zerolinecolor: "rgba(255,255,255,0.08)",
            tickfont: { size: 9, color: '#333' },
          }
        }}
        config={{
          displayModeBar: false,
          scrollZoom: true,
          responsive: true
        }}
      />
    </div>
  );
}
