import { useState, useEffect } from 'react'
import './App.css'
import Header from './components/Header'
import UploadZone from './components/UploadZone'
import StatCard from './components/StatCard'
import TimelineBar from './components/TimelineBar'
import ThumbnailGallery from './components/ThumbnailGallery'
import VideoPlayer from './components/VideoPlayer'
import LiveTracker from './components/LiveTracker'

export default function App() {
  const [tab, setTab] = useState('video') // 'video' or 'live'
  const [photo, setPhoto] = useState(null)
  const [video, setVideo] = useState(null)
  const [photoPreview, setPhotoPreview] = useState(null)
  const [videoPreview, setVideoPreview] = useState(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)
  const [gpuName, setGpuName] = useState(null)
  const [progressMsg, setProgressMsg] = useState('')
  const [processTime, setProcessTime] = useState(null)   // ① timer

  useEffect(() => {
    fetch('/api/health')
      .then((r) => r.json())
      .then((d) => setGpuName(d.gpu))
      .catch(() => setGpuName('CPU'))
  }, [])

  const handlePhotoSelect = (file) => {
    setPhoto(file)
    setPhotoPreview(URL.createObjectURL(file))
    setResults(null); setError(null); setProcessTime(null)
  }

  const handleVideoSelect = (file) => {
    setVideo(file)
    setVideoPreview(URL.createObjectURL(file))
    setResults(null); setError(null); setProcessTime(null)
  }

  // ④ Reset handler
  const handleReset = () => {
    setPhoto(null); setVideo(null)
    setPhotoPreview(null); setVideoPreview(null)
    setResults(null); setError(null); setProcessTime(null)
  }

  const messages = [
    '🔍 Extracting reference face embedding…',
    '🎬 Decoding video frames via NVDEC…',
    '🧠 Running InsightFace SCRFD detection…',
    '⚡ Matching faces with ArcFace on GPU…',
    '🎯 Compiling detections…',
    '🎬 Encoding tracked video with NVENC…',
  ]

  const analyze = async () => {
    if (!photo || !video) return
    setIsAnalyzing(true)
    setResults(null)
    setError(null)
    setProcessTime(null)

    let idx = 0
    setProgressMsg(messages[0])
    const interval = setInterval(() => {
      idx = (idx + 1) % messages.length
      setProgressMsg(messages[idx])
    }, 2500)

    const form = new FormData()
    form.append('photo', photo)
    form.append('video', video)

    const t0 = Date.now()                          // ① start timer
    try {
      const res = await fetch('/api/analyze_video', { method: 'POST', body: form })
      const data = await res.json()
      const elapsed = ((Date.now() - t0) / 1000).toFixed(1)
      setProcessTime(elapsed)                      // ① save elapsed
      if (!res.ok) setError(data.error || 'Analysis failed')
      else setResults(data)
    } catch {
      setError('Cannot reach Flask backend. Make sure it is running on port 5000.')
    } finally {
      clearInterval(interval)
      setIsAnalyzing(false)
    }
  }

  const fmt = (s) => {
    if (s == null) return '—'
    const m = Math.floor(s / 60)
    const sec = (s % 60).toFixed(1).padStart(4, '0')
    return `${m}:${sec}`
  }

  const { stats, timeline, thumbnails, video_url, match_found } = results || {}

  // Short GPU label for display
  const gpuShort = gpuName?.replace('NVIDIA ', '').replace(' Laptop GPU (CUDA)', '') || 'GPU'

  return (
    <div className="app">
      <Header gpuName={gpuName} />

      {/* ── Tab Switcher ───────────────────────────── */}
      <div className="tab-switcher">
        <button
          className={`tab-btn ${tab === 'video' ? 'active' : ''}`}
          onClick={() => setTab('video')}
        >
          📹 Video Analysis
        </button>
        <button
          className={`tab-btn ${tab === 'live' ? 'active' : ''}`}
          onClick={() => setTab('live')}
        >
          📷 Live Webcam Tracking
        </button>
      </div>

      {tab === 'video' ? (
        <>
          {/* ── Upload ─────────────────────────────────── */}
          <div className="upload-grid">
            <UploadZone
              label="Suspect Photo"
              icon="🧑"
              accept="image/*"
              onFileSelect={handlePhotoSelect}
              preview={photoPreview}
              fileName={photo?.name}
              isAnalyzing={isAnalyzing}
            />
            <UploadZone
              label="CCTV / Video"
              icon="📹"
              accept="video/*"
              onFileSelect={handleVideoSelect}
              preview={videoPreview}
              fileName={video?.name}
              isAnalyzing={isAnalyzing}
            />
          </div>

          {/* ── Buttons ─────────────────────────────────── */}
          <div className="btn-row">
            <button
              className="btn-analyze"
              disabled={!photo || !video || isAnalyzing}
              onClick={analyze}
            >
              {isAnalyzing ? (
                <>
                  <span style={{ display: 'inline-block', animation: 'spin 1s linear infinite' }}>⟳</span>
                  Scanning…
                </>
              ) : (
                <>🔍 Scan Video</>
              )}
            </button>

            {/* ④ Reset button — only visible after a result */}
            {results && (
              <button className="btn-reset" onClick={handleReset}>
                ↺ Scan Another
              </button>
            )}
          </div>

          {/* ── Progress ───────────────────────────────── */}
          {isAnalyzing && (
            <div className="progress-wrap">
              <div className="progress-bar-outer">
                <div className="progress-bar-inner" />
              </div>
              <p className="progress-label">{progressMsg}</p>
            </div>
          )}

          {/* ── Error ──────────────────────────────────── */}
          {error && (
            <div className="error-box">
              ⚠️ {error}
            </div>
          )}

          {/* ── Results ────────────────────────────────── */}
          {results && (
            <>
              {/* ① Processing Timer */}
              {processTime && (
                <div className="timer-badge">
                  ⚡ Processed in <strong>{processTime}s</strong> on {gpuShort}
                </div>
              )}

              {/* Alert Banner */}
              <div className={`alert-banner ${match_found ? 'match' : 'no-match'}`}>
                <span className="banner-icon">{match_found ? '🚨' : '❌'}</span>
                <div>
                  <div className={`banner-title ${match_found ? 'green' : 'red'}`}>
                    {match_found ? 'SUSPECT LOCATED' : 'SUSPECT NOT FOUND'}
                  </div>
                  <div className="banner-sub">
                    {match_found
                      ? `Identified in ${stats.total_detections} frames · Avg confidence ${stats.avg_confidence}% · GPU: ${stats.gpu}`
                      : `Scanned ${stats.frames_analyzed} frames across ${stats.video_duration}s of footage`}
                  </div>
                </div>
              </div>

              {/* ⑤ No-match tips */}
              {!match_found && (
                <div className="no-match-tips">
                  <div className="tips-title">💡 Tips to improve detection</div>
                  <ul>
                    <li>Use a clear, well-lit <strong>frontal face photo</strong> of the suspect</li>
                    <li>Make sure the suspect <strong>actually appears</strong> in the video clip</li>
                    <li>Try a <strong>shorter clip</strong> focused on the relevant segment</li>
                    <li>Avoid heavily compressed or blurry reference photos</li>
                  </ul>
                </div>
              )}

              {/* Stats Cards */}
              <div className="stats-grid">
                <StatCard label="Detections" value={stats.total_detections} unit="frames matched" delay={0} />
                <StatCard label="First Seen" value={fmt(stats.first_seen)} unit="timestamp" delay={0.07} />
                <StatCard label="Last Seen" value={fmt(stats.last_seen)} unit="timestamp" delay={0.14} />
                <StatCard label="Avg Confidence" value={stats.avg_confidence ? `${stats.avg_confidence}%` : '—'} unit="match score" delay={0.21} />
                <StatCard label="Duration" value={`${stats.video_duration}s`} unit="video length" delay={0.28} />
              </div>

              {/* Timeline */}
              <TimelineBar timeline={timeline} duration={stats.video_duration} />

              {/* Thumbnails */}
              <ThumbnailGallery thumbnails={thumbnails} />

              {/* Video Player */}
              <VideoPlayer videoUrl={video_url} />
            </>
          )}
        </>
      ) : (
        <div className="live-mode-wrap">
          <div className="live-upload-row">
            <UploadZone
              label="Suspect Photo"
              icon="🧑"
              accept="image/*"
              onFileSelect={handlePhotoSelect}
              preview={photoPreview}
              fileName={photo?.name}
              isAnalyzing={false}
            />
            <div className="live-instructions">
              <h3>Live Tracker Instructions</h3>
              <ul>
                <li>1. Upload a <strong>frontal photo</strong> of the target person.</li>
                <li>2. Connect your webcam and click <strong>Start Tracking</strong>.</li>
                <li>3. The system will look for this person in the live stream.</li>
              </ul>
            </div>
          </div>
          <LiveTracker photo={photo} />
        </div>
      )}

      <style>{`
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
      `}</style>
    </div>
  )
}
