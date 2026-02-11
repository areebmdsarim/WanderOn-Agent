import React from 'react';
import { Sliders, Thermometer, Database, Cpu } from 'lucide-react';

export default function ConfigPanel({ config, setConfig }) {
    const handleChange = (key, value) => {
        setConfig({ ...config, [key]: value });
    };

    return (
        <div className="flex flex-col gap-6 p-6 overflow-y-auto h-full scrollbar-none">
            <div className="flex items-center gap-4 text-primary font-bold border-b border-glass">
                <Sliders size={18} />
                <span className="text-lg tracking-tight">Model Tuning</span>
            </div>

            <div className="flex flex-col gap-4">
                {/* Model Selection Card */}
                <div className="glass-card p-4 space-y-3 border border-glass/30 hover:border-primary/30 transition-colors">
                    <div className="flex items-center gap-2 text-text-dim">
                        <Cpu size={14} className="text-accent" />
                        <span className="text-xs font-bold uppercase tracking-wider">Computation Model</span>
                    </div>
                    <select
                        value={config.model}
                        onChange={(e) => handleChange('model', e.target.value)}
                        className="w-full bg-bg-deep border border-glass rounded-lg px-3 py-2 text-sm text-text-main focus:outline-none focus:border-primary transition-all"
                    >
                        <option value="llama3.1:8b">Llama 3.1 8B</option>
                        <option value="mistral">Mistral</option>
                        <option value="phi3">Phi-3</option>
                    </select>
                </div>

                {/* Temperature Adjustment Card */}
                <div className="glass-card p-4 space-y-4 border border-glass/30 hover:border-primary/30 transition-colors">
                    <div className="flex justify-between items-center">
                        <div className="flex items-center gap-2 text-text-dim">
                            <Thermometer size={14} className="text-warning" />
                            <span className="text-xs font-bold uppercase tracking-wider">Temperature</span>
                        </div>
                        <span className="text-primary font-mono text-sm font-bold bg-primary/10 px-2 py-0.5 rounded">{config.temperature}</span>
                    </div>
                    <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.1"
                        value={config.temperature}
                        onChange={(e) => handleChange('temperature', parseFloat(e.target.value))}
                        className="w-full h-1.5 bg-bg-deep rounded-lg appearance-none cursor-pointer accent-primary border border-glass"
                    />
                    <div className="flex justify-between text-[10px] text-text-muted font-medium uppercase px-0.5">
                        <span>Precise</span>
                        <span>Creative</span>
                    </div>
                </div>

                {/* Top K Chunks Card */}
                <div className="glass-card p-4 space-y-4 border border-glass/30 hover:border-primary/30 transition-colors">
                    <div className="flex justify-between items-center">
                        <div className="flex items-center gap-2 text-text-dim">
                            <Database size={14} className="text-success" />
                            <span className="text-xs font-bold uppercase tracking-wider">RAG Precision</span>
                        </div>
                        <span className="text-primary font-mono text-sm font-bold bg-primary/10 px-2 py-0.5 rounded">{config.top_k}</span>
                    </div>
                    <input
                        type="range"
                        min="1"
                        max="10"
                        step="1"
                        value={config.top_k}
                        onChange={(e) => handleChange('top_k', parseInt(e.target.value))}
                        className="w-full h-1.5 bg-bg-deep rounded-lg appearance-none cursor-pointer accent-primary border border-glass"
                    />
                    <div className="flex justify-between text-[10px] text-text-muted font-medium uppercase px-0.5">
                        <span>Focused</span>
                        <span>Deep</span>
                    </div>
                </div>
            </div>
        </div>
    );
}
