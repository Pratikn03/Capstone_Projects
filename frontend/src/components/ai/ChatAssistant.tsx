'use client';

import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useChat } from 'ai/react';
import { SendIcon, BotIcon, XIcon, Sparkles, ChevronDown } from 'lucide-react';

export function ChatAssistant() {
  const [isOpen, setIsOpen] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const { messages, input, setInput, handleSubmit, isLoading, error } = useChat({
    api: '/api/chat',
  });

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    if (isOpen) inputRef.current?.focus();
  }, [isOpen]);

  const quickPrompts = [
    'Show load forecast with 90% PI',
    'Optimize dispatch for lowest carbon',
    'Battery SOC schedule',
    'Grid status overview',
    'Cost vs carbon tradeoff',
    'Model info and training details',
  ];

  return (
    <>
      {/* Floating Trigger Button */}
      <motion.button
        className="fixed bottom-6 right-6 h-14 w-14 bg-energy-primary rounded-full shadow-2xl z-50 flex items-center justify-center text-grid-dark hover:brightness-110 transition-all glow-primary"
        whileHover={{ scale: 1.08 }}
        whileTap={{ scale: 0.92 }}
        onClick={() => setIsOpen(!isOpen)}
      >
        <AnimatePresence mode="wait">
          {isOpen ? (
            <motion.div key="close" initial={{ rotate: -90 }} animate={{ rotate: 0 }} exit={{ rotate: 90 }}>
              <ChevronDown className="w-6 h-6" />
            </motion.div>
          ) : (
            <motion.div key="open" initial={{ scale: 0 }} animate={{ scale: 1 }} exit={{ scale: 0 }}>
              <Sparkles className="w-6 h-6" />
            </motion.div>
          )}
        </AnimatePresence>
      </motion.button>

      {/* Chat Window */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: 40, scale: 0.92 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 40, scale: 0.92 }}
            transition={{ type: 'spring', damping: 25, stiffness: 300 }}
            className="fixed bottom-24 right-6 w-[460px] h-[620px] glass-panel rounded-2xl flex flex-col overflow-hidden z-40 border border-white/10"
          >
            {/* Header */}
            <div className="px-5 py-3.5 border-b border-white/8 flex justify-between items-center bg-white/3">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-lg bg-energy-primary/20 flex items-center justify-center">
                  <BotIcon className="w-4 h-4 text-energy-primary" />
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-white">Query Assistant</h3>
                  <span className="text-[10px] text-slate-500">Natural language grid control</span>
                </div>
              </div>
              <button onClick={() => setIsOpen(false)} className="p-1.5 rounded-lg hover:bg-white/10 transition-colors">
                <XIcon className="w-4 h-4 text-slate-400" />
              </button>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
              {messages.length === 0 && (
                <div className="text-center py-6">
                  <Sparkles className="w-8 h-8 text-energy-primary/40 mx-auto mb-3" />
                  <p className="text-sm text-slate-400 mb-4">
                    Ask me to forecast, optimize, or analyze your grid.
                  </p>
                  <div className="grid grid-cols-1 gap-1.5">
                    {quickPrompts.map((prompt) => (
                      <button
                        key={prompt}
                        onClick={() => setInput(prompt)}
                        className="text-left text-xs text-slate-400 px-3 py-2 rounded-lg bg-white/3 hover:bg-white/6 hover:text-slate-200 transition-all border border-white/5"
                      >
                        {prompt}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {messages.map((msg) => (
                <div key={msg.id} className="w-full">
                  {msg.role === 'user' ? (
                    <div className="flex justify-end">
                      <div className="max-w-[85%] px-3.5 py-2.5 rounded-2xl rounded-br-md bg-energy-primary/20 border border-energy-primary/20 text-sm text-slate-100">
                        {msg.content}
                      </div>
                    </div>
                  ) : (
                    <div className="flex gap-2.5">
                      <div className="w-6 h-6 rounded-full bg-energy-primary/15 flex items-center justify-center flex-shrink-0 mt-0.5">
                        <BotIcon className="w-3 h-3 text-energy-primary" />
                      </div>
                      <div className="flex-1 min-w-0">
                        {msg.content && (
                          <div className="text-sm text-slate-200 leading-relaxed whitespace-pre-wrap">
                            {msg.content}
                          </div>
                        )}
                        {msg.toolInvocations?.map((invocation: any, i: number) => (
                          <div key={i} className="mt-2 glass-panel rounded-lg p-3 text-xs">
                            <div className="flex items-center gap-1.5 mb-1.5">
                              <span className="w-1.5 h-1.5 rounded-full bg-energy-primary" />
                              <span className="text-energy-primary font-medium">
                                {invocation.toolName?.replace(/_/g, ' ')}
                              </span>
                            </div>
                            {invocation.state === 'result' && (
                              <pre className="text-slate-400 overflow-x-auto text-[10px] leading-relaxed max-h-40 overflow-y-auto">
                                {JSON.stringify(invocation.result, null, 2)}
                              </pre>
                            )}
                            {invocation.state === 'call' && (
                              <div className="text-slate-500 animate-pulse">Processing...</div>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ))}

              {isLoading && (
                <div className="flex items-center gap-2.5">
                  <div className="w-6 h-6 rounded-full bg-energy-primary/15 flex items-center justify-center flex-shrink-0">
                    <BotIcon className="w-3 h-3 text-energy-primary" />
                  </div>
                  <div className="flex items-center gap-2 text-xs text-slate-400">
                    <span className="flex gap-0.5">
                      <span className="w-1.5 h-1.5 rounded-full bg-energy-primary animate-bounce" style={{ animationDelay: '0ms' }} />
                      <span className="w-1.5 h-1.5 rounded-full bg-energy-primary animate-bounce" style={{ animationDelay: '150ms' }} />
                      <span className="w-1.5 h-1.5 rounded-full bg-energy-primary animate-bounce" style={{ animationDelay: '300ms' }} />
                    </span>
                    Analyzing...
                  </div>
                </div>
              )}

              {error && (
                <div className="text-xs text-energy-alert px-3 py-2 rounded-lg bg-energy-alert/10 border border-energy-alert/20">
                  {error.message || 'Connection error — check OPENAI_API_KEY in .env.local'}
                </div>
              )}

              <div ref={scrollRef} />
            </div>

            {/* Input */}
            <div className="px-4 py-3 border-t border-white/8 bg-white/2">
              <form onSubmit={handleSubmit} className="relative">
                <input
                  ref={inputRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  disabled={isLoading}
                  className="w-full glass-input rounded-xl py-3 px-4 pr-12 text-sm text-slate-100 placeholder:text-slate-600 disabled:opacity-50"
                  placeholder="Optimize dispatch, forecast load..."
                />
                <button
                  type="submit"
                  disabled={!input.trim() || isLoading}
                  className="absolute right-2 top-1/2 -translate-y-1/2 p-2 rounded-lg bg-energy-primary/20 hover:bg-energy-primary/30 disabled:opacity-30 transition-all"
                >
                  <SendIcon className="w-4 h-4 text-energy-primary" />
                </button>
              </form>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
