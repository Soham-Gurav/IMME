"use client";
import { useState } from "react";
import axios from "axios";
import { BASE_URL } from "@/lib/api";

export default function Upload() {
  const [file, setFile] = useState<any>(null);
  const [results, setResults] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);

    const formData = new FormData();
    formData.append("image", file);

    try {
      const res = await axios.post(`${BASE_URL}/upload`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResults(res.data);
    } catch (err) {
      console.error("Upload error:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-10">
      {/* Upload Zone */}
      <div className="flex flex-col items-center justify-center">
        <label className="group relative w-full max-w-xl flex flex-col items-center justify-center h-48 border-2 border-dashed border-white/10 rounded-[2rem] bg-white/[0.02] hover:bg-white/[0.04] hover:border-white/20 transition-all cursor-pointer overflow-hidden">
          <div className="flex flex-col items-center justify-center pt-5 pb-6">
            <svg className="w-8 h-8 mb-4 text-gray-500 group-hover:text-blue-500 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4" />
            </svg>
            <p className="mb-2 text-sm text-gray-400 font-medium">
              {file ? file.name : "Drop image or click to browse"}
            </p>
            <p className="text-[10px] text-gray-600 uppercase tracking-widest">Supports JPG, PNG, WEBP</p>
          </div>
          <input 
            type="file" 
            className="hidden" 
            onChange={(e) => setFile(e.target.files?.[0])} 
          />
        </label>

        <button
          onClick={handleUpload}
          disabled={!file || loading}
          className={`mt-8 px-12 py-4 rounded-full font-semibold transition-all duration-300 ${
            !file || loading 
              ? "bg-white/5 text-gray-600 cursor-not-allowed" 
              : "bg-white text-black hover:scale-105 active:scale-95 shadow-xl shadow-white/5"
          }`}
        >
          {loading ? (
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 border-2 border-black/20 border-t-black rounded-full animate-spin" />
              Processing...
            </div>
          ) : "Analyze Latent Space"}
        </button>
      </div>

      {/* Analysis Results */}
      {results.length > 0 && (
        <div className="space-y-4 animate-in fade-in slide-in-from-bottom-4 duration-1000">
          <h3 className="text-[10px] font-bold uppercase tracking-[0.3em] text-gray-600 px-2">Similarity Scores</h3>
          <div className="grid gap-3">
            {results.map((r, i) => (
              <div 
                key={i} 
                className="flex items-center justify-between p-5 bg-white/[0.03] border border-white/[0.05] rounded-2xl group hover:border-white/20 transition-colors"
              >
                <span className="text-sm font-medium text-gray-300 group-hover:text-white transition-colors">
                  {r.caption}
                </span>
                <div className="flex items-center gap-4">
                  <div className="h-1.5 w-24 bg-white/5 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-blue-500 rounded-full" 
                      style={{ width: `${r.score * 100}%` }}
                    />
                  </div>
                  <span className="text-sm font-mono text-blue-500 w-12 text-right">
                    {r.score.toFixed(3)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
