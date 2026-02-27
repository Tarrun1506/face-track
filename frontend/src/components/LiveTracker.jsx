import { useRef, useEffect, useState } from 'react'

export default function LiveTracker({ photo, onPhotoCapture }) {
    const videoRef = useRef(null)
    const canvasRef = useRef(null)
    const [isTracking, setIsTracking] = useState(false)
    const [webcamActive, setWebcamActive] = useState(false)
    const [error, setError] = useState(null)
    const [status, setStatus] = useState('Ready')

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
                setStatus('Reference Ready')
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

                // Draw bounding box
                ctx.strokeStyle = '#22ff88'
                ctx.lineWidth = 3
                ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)

                // Draw label
                ctx.fillStyle = '#22ff88'
                ctx.font = 'bold 14px JetBrains Mono, monospace'
                const label = `SUSPECT MATCHed: ${conf.toFixed(1)}%`
                const textWidth = ctx.measureText(label).width
                ctx.fillRect(x1, y1 - 25, textWidth + 10, 25)
                ctx.fillStyle = '#000'
                ctx.fillText(label, x1 + 5, y1 - 7)

                setStatus(`In Frame (${conf.toFixed(0)}%)`)
            } else {
                setStatus('Scanning...')
            }
        } catch (err) {
            console.error(err)
        }
    }

    return (
        <div className="live-tracker">
            <div className="live-controls">
                {!isTracking && !webcamActive && (
                    <button className="btn-analyze btn-webcam" onClick={() => setWebcamActive(true)}>
                        📷 Open Webcam
                    </button>
                )}

                {webcamActive && !isTracking && (
                    <button className="btn-analyze btn-capture" onClick={capturePhoto}>
                        📸 Take Reference Photo
                    </button>
                )}

                <button
                    className={`btn-analyze ${isTracking ? 'btn-stop' : ''}`}
                    disabled={!photo}
                    onClick={() => {
                        setIsTracking(!isTracking)
                        setWebcamActive(false)
                    }}
                >
                    {isTracking ? '🛑 Stop Live Tracking' : '📹 Start Live Tracking'}
                </button>

                <div className={`live-status-badge ${isTracking ? 'active' : ''}`}>
                    <span className="dot" /> {status}
                </div>
            </div>

            {error && <div className="error-box">⚠️ {error}</div>}

            <div className="video-container">
                <video ref={videoRef} autoPlay playsInline muted className="live-video" />
                <canvas ref={canvasRef} width={640} height={480} className="live-canvas" />

                {!isTracking && !webcamActive && (
                    <div className="video-overlay">
                        <div className="camera-icon">📷</div>
                        <p>Webcam is inactive. Use "Open Webcam" to capture a photo or upload a suspect photo & click Start.</p>
                    </div>
                )}
            </div>
        </div>
    )
}
