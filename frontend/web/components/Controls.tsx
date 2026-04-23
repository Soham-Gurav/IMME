"use client";

export default function Controls({ method, setMethod, model, setModel }: any) {
  const methods = ["umap", "pca", "tsne"];
  const models = ["clip", "jepa"];

  return (
    <div className="flex flex-col md:flex-row items-center justify-center gap-8">
      
      {/* Method Selector */}
      <div className="flex flex-col items-center gap-3">
        <span className="text-[10px] font-bold tracking-[0.2em] text-gray-500 uppercase">
          Projection Method
        </span>
        <div className="flex p-1 bg-white/[0.03] border border-white/[0.08] rounded-full backdrop-blur-md">
          {methods.map((m) => (
            <button
              key={m}
              onClick={() => setMethod(m)}
              className={`px-6 py-2 text-sm font-medium rounded-full transition-all duration-300 ${
                method === m 
                  ? "bg-white text-black shadow-lg shadow-white/10" 
                  : "text-gray-400 hover:text-white"
              }`}
            >
              {m.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {/* Vertical Divider (Hidden on Mobile) */}
      <div className="hidden md:block w-[1px] h-10 bg-white/[0.1]" />

      {/* Model Selector */}
      <div className="flex flex-col items-center gap-3">
        <span className="text-[10px] font-bold tracking-[0.2em] text-gray-500 uppercase">
          Model Architecture
        </span>
        <div className="flex p-1 bg-white/[0.03] border border-white/[0.08] rounded-full backdrop-blur-md">
          {models.map((m) => (
            <button
              key={m}
              onClick={() => setModel(m)}
              className={`px-8 py-2 text-sm font-medium rounded-full transition-all duration-300 ${
                model === m 
                  ? "bg-white text-black shadow-lg shadow-white/10" 
                  : "text-gray-400 hover:text-white"
              }`}
            >
              {m.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

    </div>
  );
}
