import React, { useState } from 'react';

const ResultsViewer = ({ result }) => {
    if (!result || !result.summary) return null;

    const [activeTab, setActiveTab] = useState('abnormalities'); // 'abnormalities' or 'all'

    // Filter for abnormalities
    const abnormalities = result.detailed_results.filter(
        r => r.status === 'processed' && r.prediction && r.prediction.pathology !== 'Normal'
    );

    const displayList = activeTab === 'abnormalities' ? abnormalities : result.detailed_results;

    const BASE_URL = 'http://127.0.0.1:5000/';

    return (
        <div className="results-container">
            <div className="summary-card">
                <div className="summary-stat">
                    <span className="stat-value">{result.summary.total_frames_processed}</span>
                    <span className="stat-label">Frames Processed</span>
                </div>
                <div className="summary-stat">
                    <span className="stat-value text-accent">{result.summary.abnormalities_detected}</span>
                    <span className="stat-label">Abnormalities</span>
                </div>
                <div className="summary-stat">
                    <span className="stat-value text-dim">{result.summary.frames_skipped}</span>
                    <span className="stat-label">Skipped (Low Quality)</span>
                </div>
            </div>

            <div className="tabs">
                <button
                    className={`tab ${activeTab === 'abnormalities' ? 'active' : ''}`}
                    onClick={() => setActiveTab('abnormalities')}
                >
                    Detected Findings ({abnormalities.length})
                </button>
                <button
                    className={`tab ${activeTab === 'all' ? 'active' : ''}`}
                    onClick={() => setActiveTab('all')}
                >
                    All Frames ({result.detailed_results.length})
                </button>
            </div>

            <div className="findings-scroll">
                {displayList.length === 0 ? (
                    <div className="no-findings">
                        <p>{activeTab === 'abnormalities' ? 'No abnormalities detected. Great news!' : 'No frames.'}</p>
                    </div>
                ) : (
                    displayList.map((item, idx) => (
                        <div key={idx} className="finding-card">
                            <div className="finding-info">
                                <div className="finding-header">
                                    <span className={`badge ${item.status === 'skipped' ? 'badge-skipped' : (item.prediction?.pathology === 'Normal' ? 'badge-normal' : 'badge-abnormal')}`}>
                                        {item.status === 'skipped' ? 'Skipped' : item.prediction?.pathology}
                                    </span>
                                    <span className="frame-idx">Frame #{item.frame_idx}</span>
                                </div>
                                {item.status === 'processed' && (
                                    <>
                                        <div className="detail-row">
                                            <span className="label">Location:</span> {item.prediction?.anatomical}
                                        </div>
                                        <div className="detail-row">
                                            <span className="label">Confidence:</span> {(item.prediction?.pathology_confidence * 100).toFixed(1)}%
                                        </div>
                                    </>
                                )}
                                {item.status === 'skipped' && (
                                    <div className="detail-row">
                                        <span className="label">Reason:</span> {item.reason}
                                    </div>
                                )}
                            </div>

                            {item.heatmap_url && (
                                <div className="heatmap-preview">
                                    <img src={`${BASE_URL}${item.heatmap_url}`} alt="XAI Heatmap" />
                                    <span className="img-caption">Deep Care Explainable AI</span>
                                </div>
                            )}
                        </div>
                    ))
                )}
            </div>

            <style>{`
                .results-container {
                    display: flex;
                    flex-direction: column;
                    gap: 1.5rem;
                }
                .summary-card {
                    display: grid;
                    grid-template-columns: repeat(3, 1fr);
                    gap: 1rem;
                    background: rgba(255, 255, 255, 0.03);
                    padding: 1.5rem;
                    border-radius: var(--radius-md);
                    border: 1px solid var(--border);
                }
                .summary-stat {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    text-align: center;
                }
                .stat-value {
                    font-size: 2rem;
                    font-weight: 700;
                    line-height: 1;
                    margin-bottom: 0.5rem;
                }
                .stat-label {
                    font-size: 0.875rem;
                    color: var(--text-dim);
                }
                .text-accent { color: var(--accent); }
                .text-dim { color: var(--text-dim); }

                .tabs {
                    display: flex;
                    gap: 1rem;
                    border-bottom: 1px solid var(--border);
                }
                .tab {
                    background: none;
                    border: none;
                    padding: 0.75rem 1rem;
                    color: var(--text-dim);
                    font-weight: 500;
                    cursor: pointer;
                    border-bottom: 2px solid transparent;
                    transition: all 0.2s;
                }
                .tab.active {
                    color: var(--text-primary);
                    border-bottom-color: var(--primary);
                }
                
                .findings-scroll {
                    max-height: 600px;
                    overflow-y: auto;
                    display: flex;
                    flex-direction: column;
                    gap: 1rem;
                    padding-right: 0.5rem;
                }
                .finding-card {
                    display: flex;
                    justify-content: space-between;
                    background: rgba(255, 255, 255, 0.02);
                    border: 1px solid var(--border);
                    border-radius: var(--radius-sm);
                    padding: 1rem;
                    transition: transform 0.2s;
                }
                .finding-card:hover {
                    transform: translateY(-2px);
                    background: rgba(255, 255, 255, 0.04);
                }
                .finding-info {
                    flex: 1;
                    display: flex;
                    flex-direction: column;
                    gap: 0.5rem;
                }
                .finding-header {
                    display: flex;
                    align-items: center;
                    gap: 0.75rem;
                    margin-bottom: 0.25rem;
                }
                .badge {
                    padding: 0.25rem 0.6rem;
                    border-radius: 99px;
                    font-size: 0.75rem;
                    font-weight: 600;
                    text-transform: uppercase;
                    letter-spacing: 0.05em;
                }
                .badge-abnormal { background: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); }
                .badge-normal { background: rgba(34, 197, 94, 0.2); color: #22c55e; border: 1px solid rgba(34, 197, 94, 0.3); }
                .badge-skipped { background: rgba(148, 163, 184, 0.2); color: #94a3b8; border: 1px solid rgba(148, 163, 184, 0.3); }
                
                .frame-idx { color: var(--text-dim); font-size: 0.8rem; font-family: monospace; }
                
                .detail-row {
                    font-size: 0.9rem;
                    color: var(--text-secondary);
                }
                .label { color: var(--text-dim); margin-right: 0.5rem; }

                .heatmap-preview {
                    width: 120px;
                    height: 120px;
                    border-radius: var(--radius-sm);
                    overflow: hidden;
                    position: relative;
                    flex-shrink: 0;
                    margin-left: 1rem;
                }
                .heatmap-preview img {
                    width: 100%;
                    height: 100%;
                    object-fit: cover;
                }
                .img-caption {
                    position: absolute;
                    bottom: 0;
                    left: 0;
                    right: 0;
                    background: rgba(0,0,0,0.7);
                    color: white;
                    font-size: 0.6rem;
                    text-align: center;
                    padding: 2px;
                }
                .no-findings {
                    text-align: center;
                    padding: 2rem;
                    color: var(--text-dim);
                    border: 1px dashed var(--border);
                    border-radius: var(--radius-md);
                }
            `}</style>
        </div>
    );
};

export default ResultsViewer;
