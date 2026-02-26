export default function TimelineBar({ timeline, duration }) {
    if (!timeline || timeline.length === 0) return null

    // Group into max 200 buckets for visual clarity
    const BUCKETS = Math.min(200, timeline.length)
    const chunkSize = Math.ceil(timeline.length / BUCKETS)
    const buckets = []

    for (let i = 0; i < timeline.length; i += chunkSize) {
        const chunk = timeline.slice(i, i + chunkSize)
        const detected = chunk.some((t) => t.detected)
        buckets.push(detected)
    }

    const fmt = (s) => {
        const m = Math.floor(s / 60)
        const sec = (s % 60).toFixed(0).padStart(2, '0')
        return `${m}:${sec}`
    }

    return (
        <div className="timeline-section">
            <div className="section-title">Detection Timeline</div>
            <div className="timeline-bar">
                {buckets.map((det, i) => (
                    <div key={i} className={`tl-seg ${det ? 'detected' : 'undetected'}`} />
                ))}
            </div>
            <div className="tl-labels">
                <span className="tl-label">0:00</span>
                <span className="tl-label" style={{ color: 'var(--green)', fontSize: '0.7rem' }}>
                    ▬ Suspect Present
                </span>
                <span className="tl-label">{fmt(duration || 0)}</span>
            </div>
        </div>
    )
}
