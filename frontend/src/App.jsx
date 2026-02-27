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
      <div className="tab-switcher" style={{ marginBottom: '40px' }}>
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
          📷 Live Tracking
        </button>
      </div>

      {tab === 'video' ? (
        <div className="video-mode-wrap" style={{ animation: 'fadeIn 0.5s ease' }}>
          {/* ── Upload ─────────────────────────────────── */}
          <div className="upload-grid">
            <UploadZone
              label="Suspect Reference"
              icon="🧑"
              accept="image/*"
              onFileSelect={handlePhotoSelect}
              preview={photoPreview}
              fileName={photo?.name}
              isAnalyzing={isAnalyzing}
            />
            <UploadZone
              label="Surveillance Footage"
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
                  <span style={{ display: 'inline-block', animation: 'spin 1s linear infinite', marginRight: '8px' }}>⟳</span>
                  Analyzing...
                </>
              ) : (
                <>🔍 Execute Analysis</>
              )}
            </button>

            {results && (
              <button className="btn-reset" onClick={handleReset}>
                ↺ Reset Dashboard
              </button>
            )}
          </div>

          {/* ── Progress ───────────────────────────────── */}
          {isAnalyzing && (
            <div className="progress-wrap glass-panel" style={{ marginBottom: '40px' }}>
              <div className="progress-bar-outer">
                <div className="progress-bar-inner" />
              </div>
              <p className="progress-label">{progressMsg}</p>
            </div>
          )}

          {/* ── Error ──────────────────────────────────── */}
          {error && (
            <div className="error-box glass-panel">
              ⚠️ {error}
            </div>
          )}

          {/* ── Results ────────────────────────────────── */}
          {results && (
            <div className="results-container" style={{ animation: 'fadeIn 0.5s ease' }}>
              {processTime && (
                <div className="timer-badge">
                  ⚡ Computational Latency: <strong>{processTime}s</strong> on {gpuShort}
                </div>
              )}

              <div className={`alert-banner ${match_found ? 'match' : 'no-match'}`}>
                <span className="banner-icon">{match_found ? '🚨' : '❌'}</span>
                <div>
                  <div className={`banner-title ${match_found ? 'green' : 'red'}`}>
                    {match_found ? 'TARGET IDENTIFIED' : 'NO TARGET MATCH'}
                  </div>
                  <div className="banner-sub">
                    {match_found
                      ? `Positive ID in ${stats.total_detections} frames · Confidence ${stats.avg_confidence}% · ${stats.gpu}`
                      : `Scanned ${stats.frames_analyzed} frames · ${stats.video_duration}s duration`}
                  </div>
                </div>
              </div>

              {!match_found && (
                <div className="no-match-tips glass-panel">
                  <div className="tips-title">💡 Analysis Recommendations</div>
                  <ul>
                    <li>Use a high-resolution <strong>frontal reference</strong></li>
                    <li>Ensure target face is <strong>visible</strong> in footage</li>
                    <li>Adjust <strong>lighting conditions</strong> for reference capture</li>
                  </ul>
                </div>
              )}

              <div className="stats-grid">
                <StatCard label="Matches" value={stats.total_detections} unit="frames" delay={0} />
                <StatCard label="Entry" value={fmt(stats.first_seen)} unit="sec" delay={0.07} />
                <StatCard label="Exit" value={fmt(stats.last_seen)} unit="sec" delay={0.14} />
                <StatCard label="Max Conf" value={stats.avg_confidence ? `${stats.avg_confidence}%` : '—'} unit="score" delay={0.21} />
                <StatCard label="Clip Len" value={`${stats.video_duration}s`} unit="total" delay={0.28} />
              </div>

              <div className="glass-panel" style={{ padding: '24px', marginBottom: '32px' }}>
                <TimelineBar timeline={timeline} duration={stats.video_duration} />
              </div>

              <div className="glass-panel" style={{ padding: '24px', marginBottom: '32px' }}>
                <ThumbnailGallery thumbnails={thumbnails} />
              </div>

              <div className="glass-panel" style={{ overflow: 'hidden' }}>
                <VideoPlayer videoUrl={video_url} />
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="live-mode-wrap" style={{ animation: 'fadeIn 0.5s ease' }}>
          <div className="live-upload-row">
            <UploadZone
              label="Suspect Target"
              icon="🧑"
              accept="image/*"
              onFileSelect={handlePhotoSelect}
              preview={photoPreview}
              fileName={photo?.name}
              isAnalyzing={false}
            />
            <div className="live-instructions glass-panel">
              <h3>Live Surveillance Protocol</h3>
              <ul>
                <li><span>1.</span><span>Establish a <strong>clear reference</strong> capture.</span></li>
                <li><span>2.</span><span>Initialize the <strong>optical sensor</strong> stream.</span></li>
                <li><span>3.</span><span>Execute <strong>real-time matching</strong> algorithm.</span></li>
              </ul>
            </div>
          </div>
          <div className="glass-panel" style={{ padding: '32px' }}>
            <LiveTracker photo={photo} onPhotoCapture={handlePhotoSelect} />
          </div>
        </div>
      )}

      <style>{`
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        .green { color: var(--primary); }
        .red { color: var(--danger); }
      `}</style>
    </div>
  )
}
