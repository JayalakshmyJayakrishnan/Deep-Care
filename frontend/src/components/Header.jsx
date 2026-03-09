import React from 'react';

const Header = () => {
    return (
        <header style={{
            borderBottom: '1px solid rgba(255,255,255,0.1)',
            padding: '1rem 2rem',
            background: 'rgba(15, 23, 42, 0.8)',
            backdropFilter: 'blur(10px)',
            position: 'sticky',
            top: 0,
            zIndex: 10
        }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                    <div style={{
                        width: '32px',
                        height: '32px',
                        background: 'var(--primary)',
                        borderRadius: '50%',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontWeight: 'bold'
                    }}>P</div>
                    <h1 style={{ fontSize: '1.25rem', margin: 0 }}>PillCam AI <span style={{ color: 'var(--text-dim)', fontWeight: 400 }}>Diagnostics</span></h1>
                </div>

                <nav>
                    <button className="btn-outline" style={{ marginRight: '1rem' }}>Docs</button>
                    <button className="btn">System Status: Online</button>
                </nav>
            </div>
        </header>
    );
};

export default Header;
