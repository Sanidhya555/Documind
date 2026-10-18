import React from 'react';

export const StatCard = ({ title, value, subtext, icon: Icon, color = 'blue' }) => {
  const colorMap = {
    blue: {
      bg: 'bg-blue-500/10 border-blue-500/20 text-blue-400',
      iconBg: 'bg-blue-500/20 text-blue-400',
    },
    emerald: {
      bg: 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400',
      iconBg: 'bg-emerald-500/20 text-emerald-400',
    },
    purple: {
      bg: 'bg-purple-500/10 border-purple-500/20 text-purple-400',
      iconBg: 'bg-purple-500/20 text-purple-400',
    },
    amber: {
      bg: 'bg-amber-500/10 border-amber-500/20 text-amber-400',
      iconBg: 'bg-amber-500/20 text-amber-400',
    },
    cyan: {
      bg: 'bg-cyan-500/10 border-cyan-500/20 text-cyan-400',
      iconBg: 'bg-cyan-500/20 text-cyan-400',
    },
  };

  const scheme = colorMap[color] || colorMap.blue;

  return (
    <div className={`p-5 rounded-2xl border ${scheme.bg} flex items-center justify-between transition-all duration-200 hover:scale-[1.01]`}>
      <div>
        <p className="text-xs uppercase tracking-wider text-slate-400 font-medium mb-1">{title}</p>
        <p className="text-2xl font-bold text-white tracking-tight">{value ?? '—'}</p>
        {subtext && <p className="text-xs text-slate-400 mt-1">{subtext}</p>}
      </div>
      {Icon && (
        <div className={`p-3 rounded-xl ${scheme.iconBg}`}>
          <Icon className="w-6 h-6" />
        </div>
      )}
    </div>
  );
};
