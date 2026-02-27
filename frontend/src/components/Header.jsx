export default function Header({ gpuName }) {
    return (
        <header className="header">
            <div className="header-logo">
                <div className="header-icon">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <circle cx="12" cy="12" r="10" /><circle cx="12" cy="12" r="3" /><path d="m19 19-3-3" />
                    </svg>
                </div>
                <div>
                    <h1>CCTV <span>Suspect</span> Tracker Pro</h1>
                    <p className="header-sub">Advanced Face Recognition Engine</p>
                </div>
            </div>
            <div className="gpu-badge">
                <div className="dot" />
                {gpuName || 'System Ready'}
            </div>
        </header>
    )
}
