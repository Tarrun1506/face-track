const FLASK_URL = 'http://localhost:5000'

export default function VideoPlayer({ videoUrl }) {
    if (!videoUrl) return null

    const directUrl = `${FLASK_URL}${videoUrl}`

    const handleDownload = async () => {
        try {
            const res = await fetch(directUrl)
            const blob = await res.blob()
            const url = URL.createObjectURL(blob)
            const a = document.createElement('a')
            a.href = url
            a.download = 'tracked_suspect.mp4'
            document.body.appendChild(a)
            a.click()
            document.body.removeChild(a)
            URL.revokeObjectURL(url)
        } catch {
            // Fallback: open in new tab
            window.open(directUrl, '_blank')
        }
    }

    return (
        <div className="video-section">
            <div className="section-title">Tracked Output Video</div>
            <div className="video-wrap">
                <video controls autoPlay loop src={directUrl} />
                <div className="video-controls">
                    <button className="btn-download" onClick={handleDownload}>
                        ⬇ Download Tracked Video
                    </button>
                </div>
            </div>
        </div>
    )
}
