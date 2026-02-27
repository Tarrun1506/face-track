import { useState } from 'react'

export default function ThumbnailGallery({ thumbnails }) {
    const [lightbox, setLightbox] = useState(null)  // ⑥ lightbox state

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
                    <div
                        className="thumb-item"
                        key={i}
                        onClick={() => setLightbox(t)}
                        title="Click to enlarge"
                    >
                        <img src={`data:image/jpeg;base64,${t.data}`} alt={`Detection ${i + 1}`} />
                        <span className="thumb-ts">{fmt(t.timestamp)}</span>
                        <div className="thumb-badge">{t.confidence.toFixed(1)}%</div>
                        <div className="thumb-expand">⤢</div>
                    </div>
                ))}
            </div>

            {/* ⑥ Lightbox */}
            {lightbox && (
                <div className="lightbox-overlay" onClick={() => setLightbox(null)}>
                    <div className="lightbox-box" onClick={(e) => e.stopPropagation()}>
                        <button className="lightbox-close" onClick={() => setLightbox(null)}>✕</button>
                        <img
                            src={`data:image/jpeg;base64,${lightbox.data}`}
                            alt="Detection fullscreen"
                            className="lightbox-img"
                        />
                        <div className="lightbox-meta">
                            <span>🕐 {fmt(lightbox.timestamp)}</span>
                            <span className="lightbox-conf">✅ {lightbox.confidence.toFixed(1)}% match</span>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}
