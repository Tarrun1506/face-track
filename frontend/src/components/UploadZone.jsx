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
            className={`upload-zone ${dragging ? 'drag-over' : ''} ${preview ? 'has-file' : ''}`}
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
        >
            {isAnalyzing && preview && <div className="scan-line" />}
            <input ref={inputRef} type="file" accept={accept}
                onChange={(e) => handleFile(e.target.files[0])} />

            <div className="zone-label">
                {icon} {label}
                <span className="badge">{accept.includes('image') ? 'JPG / PNG' : 'MP4 / MOV / AVI'}</span>
            </div>

            {!preview ? (
                <div className="zone-placeholder">
                    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                        <path strokeLinecap="round" strokeLinejoin="round"
                            d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
                    </svg>
                    <p>Drag & drop or click to upload</p>
                </div>
            ) : accept.includes('image') ? (
                <>
                    <img className="photo-preview" src={preview} alt="preview" />
                    <p className="file-name">✓ {fileName}</p>
                </>
            ) : (
                <>
                    <video className="video-preview" src={preview} controls />
                    <p className="file-name">✓ {fileName}</p>
                </>
            )}
        </div>
    )
}
