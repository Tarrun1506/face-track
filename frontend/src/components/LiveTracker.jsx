import { useRef, useEffect, useState } from 'react'

export default function LiveTracker({ photo, onPhotoCapture }) {
    const videoRef = useRef(null)
    const canvasRef = useRef(null)
    const [isTracking, setIsTracking] = useState(false)
    const [webcamActive, setWebcamActive] = useState(false)
    const [error, setError] = useState(null)
    const [status, setStatus] = useState('System Idle')

    // Prepare backend when photo changes
    useEffect(() => {
        if (!photo) return
        const prepare = async () => {
            const form = new FormData()
            form.append('photo', photo)
            try {
                const res = await fetch('/api/prepare_live', { method: 'POST', body: form })
                if (!res.ok) {
                    const data = await res.json()
                    throw new Error(data.error || 'Failed to prepare reference')
                }
                setStatus('Target Locked')
            } catch (err) {
                setError(err.message)
            }
        }
        prepare()
    }, [photo])

    useEffect(() => {
        let stream = null
        let interval = null

        const startWebcam = async () => {
            try {
                stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
                if (videoRef.current) {
                    videoRef.current.srcObject = stream
                }
            } catch (err) {
                setError("Webcam access denied. Make sure you're on HTTPS or localhost.")
                setWebcamActive(false)
            }
        }

        if (isTracking || webcamActive) {
            startWebcam()
            if (isTracking) {
                interval = setInterval(captureAndTrack, 250) // 4 fps for tracking
            }
        }

        return () => {
            if (stream) stream.getTracks().forEach(t => t.stop())
            if (interval) clearInterval(interval)
        }
    }, [isTracking, webcamActive])

    const capturePhoto = () => {
        if (!videoRef.current) return
        const video = videoRef.current
        const canvas = document.createElement('canvas')
        canvas.width = 640
        canvas.height = 480
        const ctx = canvas.getContext('2d')
        ctx.drawImage(video, 0, 0, 640, 480)

        canvas.toBlob((blob) => {
            if (blob && onPhotoCapture) {
                const file = new File([blob], "captured_suspect.jpg", { type: "image/jpeg" })
                onPhotoCapture(file)
                setWebcamActive(false)
            }
        }, 'image/jpeg', 0.95)
    }

    const captureAndTrack = async () => {
        if (!videoRef.current || !canvasRef.current) return

        const video = videoRef.current
        const canvas = canvasRef.current
        const ctx = canvas.getContext('2d')

        // Capture frame to a temporary small canvas
        const tempCanvas = document.createElement('canvas')
        tempCanvas.width = 640
        tempCanvas.height = 480
        const tctx = tempCanvas.getContext('2d')
        tctx.drawImage(video, 0, 0, 640, 480)

        const dataUrl = tempCanvas.toDataURL('image/jpeg', 0.7)

        try {
            const res = await fetch('/api/track_frame', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: dataUrl })
            })
            const data = await res.json()

            // Clear previous drawings
            ctx.clearRect(0, 0, canvas.width, canvas.height)

            if (data.found) {
                const [x1, y1, x2, y2] = data.box
                const conf = data.confidence

                // Modern Rect
                ctx.strokeStyle = '#10b981'
                ctx.lineWidth = 2
                ctx.setLineDash([10, 5])
                ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)
                ctx.setLineDash([])

                // Corners
                const len = 20
                ctx.lineWidth = 4
                ctx.beginPath()
                // Top Left
                ctx.moveTo(x1, y1 + len); ctx.lineTo(x1, y1); ctx.lineTo(x1 + len, y1)
                // Top Right
                ctx.moveTo(x2 - len, y1); ctx.lineTo(x2, y1); ctx.lineTo(x2, y1 + len)
                // Bottom Left
                ctx.moveTo(x1, y2 - len); ctx.lineTo(x1, y2); ctx.lineTo(x1 + len, y2)
                // Bottom Right
                ctx.moveTo(x2 - len, y2); ctx.lineTo(x2, y2); ctx.lineTo(x2, y2 - len)
                ctx.stroke()

                // Tag
                ctx.fillStyle = '#10b981'
                const label = `MATCH: ${conf.toFixed(1)}%`
                ctx.font = '700 12px JetBrains Mono'
                const tw = ctx.measureText(label).width
                ctx.fillRect(x1, y1 - 24, tw + 20, 24)
                ctx.fillStyle = '#000'
                ctx.fillText(label, x1 + 10, y1 - 8)

                setStatus(`TRACKING... (${conf.toFixed(0)}%)`)
            } else {
                setStatus('SCANNING...')
            }
        } catch (err) {
            console.error(err)
        }
    }

    return (
        <div className="live-tracker">
            <div className="live-controls" style={{ display: 'flex', gap: '12px', marginBottom: '24px', alignItems: 'center' }}>
                {!isTracking && !webcamActive && (
                    <button className="btn-analyze btn-webcam" onClick={() => setWebcamActive(true)} style={{ background: 'var(--accent)', boxShadow: '0 10px 20px var(--accent-glow)' }}>
                        📷 Initialize Webcam
                    </button>
                )}

                {webcamActive && !isTracking && (
                    <button className="btn-analyze btn-capture" onClick={capturePhoto} style={{ background: 'var(--accent)', boxShadow: '0 10px 20px var(--accent-glow)' }}>
                        📸 Capture Target
                    </button>
                )}

                <button
                    className={`btn-analyze ${isTracking ? 'btn-danger' : ''}`}
                    disabled={!photo}
                    onClick={() => {
                        setIsTracking(!isTracking)
                        setWebcamActive(false)
                    }}
                    style={isTracking ? { background: 'var(--danger)', boxShadow: '0 10px 20px var(--danger-dim)' } : {}}
                >
                    {isTracking ? '🛑 Terminate Batch' : '📹 Execute Live Scan'}
                </button>

                <div className={`gpu-badge ${isTracking ? 'active' : ''}`} style={{ marginLeft: 'auto', background: isTracking ? 'var(--primary-dim)' : 'rgba(255,255,255,0.05)', borderColor: isTracking ? 'var(--primary)' : 'var(--border)' }}>
                    <div className="dot" style={{ background: isTracking ? 'var(--primary)' : 'var(--text-dim)' }} />
                    {status}
                </div>
            </div>

            {error && <div className="error-box glass-panel">⚠️ {error}</div>}

            <div className="video-container" style={{ position: 'relative', borderRadius: '20px', overflow: 'hidden', border: '1px solid var(--border-bright)', boxShadow: 'var(--shadow-lg)' }}>
                <video ref={videoRef} autoPlay playsInline muted className="live-video" style={{ width: '100%', display: 'block' }} />
                <canvas ref={canvasRef} width={640} height={480} className="live-canvas" style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }} />

                {!isTracking && !webcamActive && (
                    <div className="video-overlay" style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)', color: 'var(--text-muted)' }}>
                        <div className="camera-icon" style={{ fontSize: '48px', marginBottom: '16px', opacity: 0.5 }}>📷</div>
                        <p style={{ fontFamily: 'var(--mono)', fontSize: '0.9rem' }}>Optical Sensor Offline</p>
                    </div>
                )}
            </div>
        </div>
    )
}
