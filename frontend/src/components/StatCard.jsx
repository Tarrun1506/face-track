export default function StatCard({ label, value, unit, delay = 0 }) {
    return (
        <div className="stat-card glass-panel" style={{ animationDelay: `${delay}s` }}>
            <div className="stat-label">{label}</div>
            <div className="stat-value">{value ?? '—'}</div>
            {unit && <div className="stat-unit">{unit}</div>}
        </div>
    )
}
