import React, { useState } from 'react';

const UploadZone = ({ onUploadComplete }) => {
    const [dragActive, setDragActive] = useState(false);

    const handleDrag = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true);
        } else if (e.type === "dragleave") {
            setDragActive(false);
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFiles(e.dataTransfer.files[0]);
        }
    };

    const handleChange = (e) => {
        e.preventDefault();
        if (e.target.files && e.target.files[0]) {
            handleFiles(e.target.files[0]);
        }
    };

    const handleFiles = (file) => {
        // In a real app, we would upload to backend here.
        // For now, we simulate a local preview.
        const url = URL.createObjectURL(file);
        onUploadComplete({ file, previewUrl: url });
    };

    return (
        <div
            className={`upload-zone ${dragActive ? 'active' : ''}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
        >
            <input
                type="file"
                id="file-upload"
                style={{ display: 'none' }}
                onChange={handleChange}
                accept="image/*,video/*"
            />

            <label htmlFor="file-upload" style={{ cursor: 'pointer', display: 'block' }}>
                <div className="icon">📂</div>
                <p className="bold">Click to upload or drag and drop</p>
                <p className="small">MP4, PNG, JPG up to 50MB</p>
            </label>

            <style>{`
        .upload-zone {
          border: 2px dashed var(--secondary);
          border-radius: var(--radius-md);
          padding: 3rem 1rem;
          text-align: center;
          transition: all 0.2s;
          background: rgba(255,255,255,0.02);
        }
        .upload-zone.active {
          border-color: var(--primary);
          background: rgba(14, 165, 233, 0.1);
        }
        .icon { font-size: 2.5rem; margin-bottom: 1rem; }
        .bold { font-weight: 500; margin-bottom: 0.25rem; }
        .small { font-size: 0.875rem; color: var(--text-dim); }
      `}</style>
        </div>
    );
};

export default UploadZone;
