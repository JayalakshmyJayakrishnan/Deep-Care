import { useState } from 'react'
import Header from './components/Header'
import UploadZone from './components/UploadZone'
import ResultsViewer from './components/ResultsViewer'
import './index.css'

function App() {
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const handleUploadSuccess = async (uploadData) => {
    setIsAnalyzing(true);
    setAnalysisResult(null);

    const formData = new FormData();
    formData.append('file', uploadData.file);

    try {
      const response = await fetch(`http://${window.location.hostname}:5000/api/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Upload failed');
      }

      const data = await response.json();

      // The backend returns results directly
      setAnalysisResult({
        summary: data.summary,
        detailed_results: data.detailed_results,
        imageUrl: uploadData.previewUrl
      });

    } catch (error) {
      console.error('Error analyzing image:', error);
      alert('Failed to analyze image. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleDemoLoad = async () => {
    setIsAnalyzing(true);
    setAnalysisResult(null);

    try {
      const response = await fetch(`http://${window.location.hostname}:5000/api/demo`, {
        method: 'POST',
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.error || 'Demo failed');
      }

      const data = await response.json();

      setAnalysisResult({
        summary: data.summary,
        detailed_results: data.detailed_results,
        imageUrl: `http://${window.location.hostname}:5000/uploads/${data.filename}` // Serve the copied demo file
      });

    } catch (error) {
      console.error('Error running demo:', error);
      alert(error.message);
    } finally {
      setIsAnalyzing(false);
    }
  };


  return (
    <div className="app-container">
      <Header />

      <main className="container">
        <div className="grid-layout">
          <section className="card upload-section">
            <h2>Upload Endoscopy Footage</h2>
            <p className="text-dim">Supported formats: JPG, PNG, MP4</p>
            <UploadZone onUploadComplete={handleUploadSuccess} />
            <div className="divider">
              <span>OR</span>
            </div>
            <button className="btn btn-secondary full-width" onClick={handleDemoLoad} disabled={isAnalyzing}>
              {isAnalyzing ? 'Loading...' : 'Run with Galar Dataset Sample'}
            </button>
          </section>

          <section className="card results-section">
            <h2>Analysis Results</h2>
            {isAnalyzing ? (
              <div className="loading-state">
                <div className="spinner"></div>
                <p>AI Engine is analyzing frames...</p>
              </div>
            ) : analysisResult ? (
              <ResultsViewer result={analysisResult} />
            ) : (
              <div className="empty-state">
                <p>No data to display. Upload a file to begin analysis.</p>
              </div>
            )}
          </section>
        </div>
      </main>

      <style>{`
        .grid-layout {
          display: grid;
          grid-template-columns: 1fr 1.5fr;
          gap: 2rem;
          margin-top: 2rem;
        }
        .text-dim { color: var(--text-dim); }
        .empty-state {
          height: 300px;
          display: flex;
          align-items: center;
          justify-content: center;
          color: var(--text-dim);
          border: 2px dashed var(--secondary);
          border-radius: var(--radius-md);
        }
        .spinner {
          border: 4px solid var(--bg-card);
          border-top: 4px solid var(--primary);
          border-radius: 50%;
          width: 40px;
          height: 40px;
          animation: spin 1s linear infinite;
          margin: 0 auto 1rem;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .loading-state { text-align: center; padding: 3rem; }
      `}</style>
    </div>
  )
}

export default App
