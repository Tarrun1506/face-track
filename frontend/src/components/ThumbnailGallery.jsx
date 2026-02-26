export default function ThumbnailGallery({ thumbnails }) {
    if (!thumbnails || thumbnails.length === 0) return null

    const fmt = (s) => {
        const m = Math.floor(s / 60)
        const sec = (s % 60).toFixed(1).padStart(4, '0')
        return `${m}:${sec}`
    }

    return (
        <div className="thumbs-section">
            <div className="section-title">Top Detection Snapshots</div>
            <div className="thumbs-grid">
                {thumbnails.map((t, i) => (
                    <div className="thumb-item" key={i}>
                        <img src={`data:image/jpeg;base64,${t.data}`} alt={`Detection ${i + 1}`} />
                        <span className="thumb-ts">{fmt(t.timestamp)}</span>
                        <div className="thumb-badge">{t.confidence.toFixed(1)}%</div>
                    </div>
                ))}
            </div>
        </div>
    )
}
