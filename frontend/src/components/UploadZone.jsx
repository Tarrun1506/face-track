import { useRef, useState } from 'react'

export default function UploadZone({ label, accept, icon, onFileSelect, preview, fileName, isAnalyzing }) {
    const inputRef = useRef()
    const [dragging, setDragging] = useState(false)

    const handleFile = (file) => {
        if (!file) return
        onFileSelect(file)
    }

    const onDrop = (e) => {
        e.preventDefault(); setDragging(false)
        const file = e.dataTransfer.files[0]
        if (file) handleFile(file)
    }

    return (
        <div
            className={`upload-zone glass-panel ${dragging ? 'drag-over' : ''} ${preview ? 'has-file' : ''}`}
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
        >
            <input ref={inputRef} type="file" accept={accept}
                onChange={(e) => handleFile(e.target.files[0])} />

            <div className="zone-label">
                <span>{icon} {label}</span>
                <span className="badge">{accept.includes('image') ? 'JPG / PNG' : 'MP4 / MOV'}</span>
            </div>

            {!preview ? (
                <div className="zone-placeholder">
                    <div className="upload-icon-circle" style={{ marginBottom: '10px' }}>
                        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="17 8 12 3 7 8" /><line x1="12" y1="3" x2="12" y2="15" />
                        </svg>
                    </div>
                    <p>Tap to upload or drop file here</p>
                </div>
            ) : accept.includes('image') ? (
                <div className="preview-wrap">
                    <img className="photo-preview" src={preview} alt="preview" />
                    <div className="file-name" style={{ marginTop: '12px', textAlign: 'center' }}>
                        <span className="check">✓</span> {fileName}
                    </div>
                </div>
            ) : (
                <div className="preview-wrap">
                    <video className="video-preview" src={preview} />
                    <div className="file-name" style={{ marginTop: '12px', textAlign: 'center' }}>
                        <span className="check">✓</span> {fileName}
                    </div>
                </div>
            )}
        </div>
    )
}
