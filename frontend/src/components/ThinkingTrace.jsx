import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { CheckCircle2, AlertCircle, Info, ShieldCheck, Route, Search, Wrench, Brain, ChevronDown, ChevronRight } from 'lucide-react';

const iconMap = {
    INPUT_GUARDRAIL: ShieldCheck,
    TOKEN_GUARDRAIL: ShieldCheck,
    ROUTING: Route,
    CONFIDENCE_GUARDRAIL: ShieldCheck,
    RAG_PIPELINE: Search,
    TOOL_EXECUTION: Wrench,
    EXECUTION: Brain,
};

function TraceStep({ step, idx }) {
    const [isOpen, setIsOpen] = useState(step.status !== 'success');
    const Icon = iconMap[step.event] || Info;

    return (
        <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: idx * 0.1 }}
            className="timeline-step relative pb-8 group"
        >
            <div className={`absolute -left-[1.5rem] top-0 w-4 h-4 rounded-full border-2 border-bg-deep z-10 transition-colors
        ${step.status === 'success' ? 'bg-success' : step.status === 'failure' ? 'bg-error' : 'bg-warning'}`}
            />

            <div className={`glass-card p-4 transition-all hover:scale-[1.01] ${isOpen ? 'ring-1 ring-primary/30' : ''}`}>
                <div
                    className="flex items-center justify-between cursor-pointer"
                    onClick={() => setIsOpen(!isOpen)}
                >
                    <div className="flex items-center gap-2.5">
                        <Icon size={16} className="text-primary flex-shrink-0" />
                        <span className="text-xs font-bold tracking-wider text-text-dim uppercase leading-none">{step.event}</span>
                        {isOpen ? <ChevronDown size={14} className="text-text-muted transition-transform" /> : <ChevronRight size={14} className="text-text-muted transition-transform" />}
                    </div>
                    <span className="text-[10px] text-text-muted font-mono leading-none">
                        {new Date(step.timestamp).toLocaleTimeString()}
                    </span>
                </div>

                <AnimatePresence>
                    {isOpen && (
                        <motion.div
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: 'auto', opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            transition={{ duration: 0.2 }}
                            className="overflow-hidden"
                        >
                            <p className="text-sm text-text-main mt-3 leading-relaxed">{step.message}</p>

                            {step.data && (
                                <div className="mt-4 bg-bg-deep/80 rounded-lg p-3 text-[11px] font-mono text-text-dim border border-glass overflow-x-auto ring-1 ring-white/5">
                                    <pre>{JSON.stringify(step.data, null, 2)}</pre>
                                </div>
                            )}
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>
        </motion.div>
    );
}

export default function ThinkingTrace({ trace }) {
    if (!trace || trace.length === 0) return (
        <div className="h-full flex flex-col items-center justify-center text-text-muted p-12 text-center">
            <div className="flex items-center gap-4 mb-6 opacity-40">
                <div className="w-12 h-12 rounded-2xl bg-surface border border-glass flex items-center justify-center">
                    <Brain size={24} />
                </div>
                <h3 className="text-white font-medium">Ready for Logic</h3>
            </div>
            <p className="text-sm max-w-[200px]">Submit a query to see the engine's neural trace unroll.</p>
        </div>
    );

    return (
        <div className="py-6 overflow-y-auto h-full scrollbar-none">
            <div className="flex items-center gap-2 text-primary font-bold mb-8 border-b border-glass pb-4 sticky top-0 bg-bg-surface z-20 px-2">
                <Brain size={18} />
                <span className="text-lg tracking-tight">Thinking Trace</span>
            </div>

            <div className="relative">
                <AnimatePresence mode="popLayout">
                    {trace.map((step, idx) => (
                        <TraceStep key={`${step.event}-${idx}`} step={step} idx={idx} />
                    ))}
                </AnimatePresence>
            </div>
        </div>
    );
}
