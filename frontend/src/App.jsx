import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Activity, Layers, Play, Crosshair, BarChart3 } from 'lucide-react';

const API_BASE = "http://localhost:8000/api";

export default function App() {
  const [metadata, setMetadata] = useState(null);
  const [currentSlice, setCurrentSlice] = useState(80);
  const [colorMode, setColorMode] = useState("grayscale");
  const [sliceData, setSliceData] = useState(null);
  const [segmentedMask, setSegmentedMask] = useState(null);
  const [metrics, setMetrics] = useState({ dice: null, iou: null });
  const [volMetrics, setVolMetrics] = useState(null);
  const [loadingVol, setLoadingVol] = useState(false);
  const [seed, setSeed] = useState(null);

  const canvasRef = useRef(null);

  useEffect(() => {
    axios.get(`${API_BASE}/metadata`).then(res => setMetadata(res.data)).catch(err => console.error(err));
  }, []);

  useEffect(() => {
    fetchSlice(currentSlice, colorMode);
    setSegmentedMask(null);
    setMetrics({ dice: null, iou: null });
  }, [currentSlice, colorMode]);

  const fetchSlice = async (idx, mode) => {
    try {
      const res = await axios.get(`${API_BASE}/slice/${idx}?mode=${mode}`);
      setSliceData(res.data);
    } catch (err) {
      console.error(err);
    }
  };

  const handleCanvasClick = async (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left) * (canvas.width / rect.width));
    const y = Math.floor((e.clientY - rect.top) * (canvas.height / rect.height));

    setSeed({ r: y, c: x });

    try {
      const res = await axios.post(`${API_BASE}/segment`, {
        slice_idx: currentSlice,
        seed_r: y,
        seed_c: x
      });
      setSegmentedMask(res.data.mask);
      setMetrics({ dice: res.data.dice, iou: res.data.iou });
    } catch (err) {
      console.error(err);
    }
  };

  const computeFullVolume = async () => {
    setLoadingVol(true);
    try {
      const res = await axios.get(`${API_BASE}/volume-metrics`);
      setVolMetrics(res.data);
    } catch (err) {
      console.error(err);
    }
    setLoadingVol(false);
  };

  useEffect(() => {
    if (!sliceData || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const baseImg = new Image();
    baseImg.src = sliceData.image;

    baseImg.onload = () => {
      canvas.width = baseImg.width;
      canvas.height = baseImg.height;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(baseImg, 0, 0);

      if (segmentedMask) {
        const maskImg = new Image();
        maskImg.src = segmentedMask;
        maskImg.onload = () => {
          ctx.globalAlpha = 0.5;
          ctx.drawImage(maskImg, 0, 0);
          ctx.globalAlpha = 1.0;
        };
      }

      if (seed) {
        ctx.fillStyle = '#10B981';
        ctx.beginPath();
        ctx.arc(seed.c, seed.r, 4, 0, 2 * Math.PI);
        ctx.fill();
      }
    };
  }, [sliceData, segmentedMask, seed]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/60 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Activity className="w-6 h-6 text-cyan-400" />
          <h1 className="text-xl font-bold tracking-wide">NeuroDelineate <span className="text-xs bg-cyan-950 text-cyan-400 border border-cyan-800 px-2 py-0.5 rounded-full uppercase ml-2">Clinical DIP Engine</span></h1>
        </div>
        {metadata && (
          <div className="text-xs text-slate-400 font-mono">
            Patient ID: <span className="text-slate-200">{metadata.patient_id}</span> | Voxel: {metadata.voxel_dimensions_mm.join('x')} mm
          </div>
        )}
      </header>

      {/* Main Viewport */}
      <main className="flex-1 grid grid-cols-1 lg:grid-cols-4 gap-6 p-6">
        
        {/* Canvas Slice Area */}
        <div className="lg:col-span-3 bg-slate-900/40 border border-slate-800 rounded-xl p-6 flex flex-col items-center justify-center relative">
          <div className="absolute top-4 left-4 flex gap-2">
            <button 
              onClick={() => setColorMode("grayscale")}
              className={`px-3 py-1 text-xs rounded-md border ${colorMode === "grayscale" ? "bg-slate-700 border-slate-500 text-white" : "border-slate-800 text-slate-400"}`}>
              Grayscale
            </button>
            <button 
              onClick={() => setColorMode("heatmap")}
              className={`px-3 py-1 text-xs rounded-md border ${colorMode === "heatmap" ? "bg-amber-950 border-amber-700 text-amber-300" : "border-slate-800 text-slate-400"}`}>
              Thermal Heatmap
            </button>
          </div>

          <p className="text-xs text-slate-400 mb-2 flex items-center gap-1">
            <Crosshair className="w-3.5 h-3.5 text-emerald-400" /> Click on the scan image to seed and segment
          </p>

          <canvas 
            ref={canvasRef} 
            onClick={handleCanvasClick}
            className="rounded-lg shadow-2xl border border-slate-800 cursor-crosshair max-h-[480px] object-contain bg-black"
          />

          {/* Slider */}
          <div className="w-full max-w-xl mt-6">
            <div className="flex justify-between text-xs font-mono text-slate-400 mb-2">
              <span>Slice: {currentSlice}</span>
              <span>Total Slices: {metadata?.total_slices || 155}</span>
            </div>
            <input 
              type="range" 
              min="0" 
              max={metadata ? metadata.total_slices - 1 : 155} 
              value={currentSlice} 
              onChange={(e) => setCurrentSlice(Number(e.target.value))}
              className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-500"
            />
          </div>
        </div>

        {/* Sidebar Analytics */}
        <div className="space-y-6">
          <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-5">
            <h2 className="text-sm font-semibold flex items-center gap-2 mb-4 text-cyan-400">
              <Layers className="w-4 h-4" /> Slice Validation Metrics
            </h2>
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-slate-950 p-4 rounded-lg border border-slate-800/80">
                <p className="text-xs text-slate-400">Dice Score</p>
                <p className="text-2xl font-bold font-mono text-slate-100">{metrics.dice !== null ? metrics.dice : "--"}</p>
              </div>
              <div className="bg-slate-950 p-4 rounded-lg border border-slate-800/80">
                <p className="text-xs text-slate-400">IoU Score</p>
                <p className="text-2xl font-bold font-mono text-slate-100">{metrics.iou !== null ? metrics.iou : "--"}</p>
              </div>
            </div>
          </div>

          <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-5">
            <h2 className="text-sm font-semibold flex items-center gap-2 mb-4 text-emerald-400">
              <BarChart3 className="w-4 h-4" /> 3D Volume Analysis
            </h2>
            <button 
              onClick={computeFullVolume} 
              disabled={loadingVol}
              className="w-full bg-cyan-600 hover:bg-cyan-500 text-white font-medium py-2 px-4 rounded-lg text-xs transition flex items-center justify-center gap-2">
              {loadingVol ? "Processing 155 Slices..." : <><Play className="w-3.5 h-3.5" /> Run 3D Volume Computation</>}
            </button>

            {volMetrics && (
              <div className="mt-4 space-y-3 font-mono text-xs">
                <div className="flex justify-between bg-slate-950 p-2.5 rounded border border-slate-800">
                  <span className="text-slate-400">Predicted Volume:</span>
                  <span className="text-cyan-400 font-bold">{volMetrics.predicted_volume_cm3} cm³</span>
                </div>
                <div className="flex justify-between bg-slate-950 p-2.5 rounded border border-slate-800">
                  <span className="text-slate-400">Ground Truth:</span>
                  <span className="text-slate-200">{volMetrics.ground_truth_volume_cm3} cm³</span>
                </div>
                <div className="flex justify-between bg-slate-950 p-2.5 rounded border border-slate-800">
                  <span className="text-slate-400">Error:</span>
                  <span className="text-emerald-400">{volMetrics.error_cm3} cm³</span>
                </div>
              </div>
            )}
          </div>
        </div>

      </main>
    </div>
  );
}