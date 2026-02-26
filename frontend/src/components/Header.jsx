export default function Header({ gpuName }) {
    return (
        <header className="header">
            <div className="header-logo">
                <div className="header-icon">🎯</div>
                <div>
                    <h1>CCTV <span>Suspect</span> Tracker</h1>
                    <p className="header-sub">Deep Learning · Face Recognition · GPU Accelerated</p>
                </div>
            </div>
            <div className="gpu-badge">
                <span className="dot" />
                {gpuName || 'Loading GPU…'}
            </div>
        </header>
    )
}
