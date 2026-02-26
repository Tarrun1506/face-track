import { useState, useEffect } from 'react'
import './App.css'
import Header from './components/Header'
import UploadZone from './components/UploadZone'
import StatCard from './components/StatCard'
import TimelineBar from './components/TimelineBar'
import ThumbnailGallery from './components/ThumbnailGallery'
import VideoPlayer from './components/VideoPlayer'

export default function App() {
  const [photo, setPhoto] = useState(null)
  const [video, setVideo] = useState(null)
  const [photoPreview, setPhotoPreview] = useState(null)
  const [videoPreview, setVideoPreview] = useState(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)
  const [gpuName, setGpuName] = useState(null)
  const [progressMsg, setProgressMsg] = useState('')

  // Fetch GPU name on load
  useEffect(() => {
    fetch('/api/health')
      .then((r) => r.json())
      .then((d) => setGpuName(d.gpu))
      .catch(() => setGpuName('CPU'))
  }, [])

  const handlePhotoSelect = (file) => {
    setPhoto(file)
    setPhotoPreview(URL.createObjectURL(file))
    setResults(null); setError(null)
  }

  const handleVideoSelect = (file) => {
    setVideo(file)
    setVideoPreview(URL.createObjectURL(file))
    setResults(null); setError(null)
  }

  const messages = [
    '🔍 Extracting reference face embedding…',
    '🎬 Reading video frames…',
    '🧠 Running MTCNN face detection…',
    '⚡ Matching with FaceNet on GPU…',
    '🎯 Compiling detections…',
    '🎬 Writing tracked output video…',
  ]

  const analyze = async () => {
    if (!photo || !video) return
    setIsAnalyzing(true)
    setResults(null)
    setError(null)

    // Cycle progress messages
    let idx = 0
    setProgressMsg(messages[0])
    const interval = setInterval(() => {
      idx = (idx + 1) % messages.length
      setProgressMsg(messages[idx])
    }, 3000)

    const form = new FormData()
    form.append('photo', photo)
    form.append('video', video)

    try {
      const res = await fetch('/api/analyze_video', { method: 'POST', body: form })
      const data = await res.json()
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

  return (
    <div className="app">
      <Header gpuName={gpuName} />

      {/* ── Upload ─────────────────────────────────── */}
      <div className="upload-grid">
        <UploadZone
          label="Suspect Photo"
          icon="🧑"
          accept="image/*"
          onFileSelect={handlePhotoSelect}
          preview={photoPreview}
          fileName={photo?.name}
        />
        <UploadZone
          label="CCTV / Video"
          icon="📹"
          accept="video/*"
          onFileSelect={handleVideoSelect}
          preview={videoPreview}
          fileName={video?.name}
        />
      </div>

      {/* ── Analyse Button ─────────────────────────── */}
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

          {/* Stats Cards */}
          <div className="stats-grid">
            <StatCard label="Detections" value={stats.total_detections} unit="frames matched" delay={0} />
            <StatCard label="First Seen" value={fmt(stats.first_seen)} unit="timestamp" delay={0.07} />
            <StatCard label="Last Seen" value={fmt(stats.last_seen)} unit="timestamp" delay={0.14} />
            <StatCard label="Avg Confidence" value={stats.avg_confidence ? `${stats.avg_confidence}%` : '—'} unit="similarity score" delay={0.21} />
          </div>

          {/* Timeline */}
          <TimelineBar timeline={timeline} duration={stats.video_duration} />

          {/* Thumbnails */}
          <ThumbnailGallery thumbnails={thumbnails} />

          {/* Video Player */}
          <VideoPlayer videoUrl={video_url} />
        </>
      )}

      <style>{`
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
      `}</style>
    </div>
  )
}
