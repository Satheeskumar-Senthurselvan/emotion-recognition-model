import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'

function App() {
  const [count, setCount] = useState(0)
  const [selectedFile, setSelectedFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState(null);

    // IMPORTANT: Use the correct backend URL
  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'; // Vite uses VITE_ prefix

  // This function needs to be defined INSIDE App
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
    setPrediction(null);
    setError(null);
    if (file) {
      setImagePreviewUrl(URL.createObjectURL(file));
    } else {
      setImagePreviewUrl(null);
    }
  };

  // This function also needs to be defined INSIDE App
  const handleSubmit = async () => {
    if (!selectedFile) {
      setError("Please select an image first.");
      return;
    }

    setLoading(true);
    setPrediction(null);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch(`${API_URL}/predict/image/`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP error! status: ${response.status} - ${errorText}`);
      }

      const data = await response.json();
      setPrediction(data);

    } catch (err) {
      console.error("Prediction failed:", err);
      setError(`Failed to get prediction: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };


   return (
    <div className="App">
      <header className="App-header">
        <h1>Emotion Recognition</h1>
        <input type="file" accept="image/*" onChange={handleFileChange} /> {/* <-- handleFileChange is used here */}
        {imagePreviewUrl && (
          <img src={imagePreviewUrl} alt="Preview" style={{ maxWidth: '300px', maxHeight: '300px', marginTop: '10px', border: '1px solid #ccc' }} />
        )}
        <button onClick={handleSubmit} disabled={!selectedFile || loading} style={{ marginTop: '20px', padding: '10px 20px', fontSize: '16px' }}>
          {loading ? 'Predicting...' : 'Predict Emotion'}
        </button>

        {error && <p style={{ color: 'red' }}>Error: {error}</p>}

        {prediction && (
          <div style={{ marginTop: '20px', textAlign: 'left' }}>
            <h2>Prediction:</h2>
            <p>Emotion: <strong>{prediction.emotion}</strong> (Confidence: {prediction.confidence.toFixed(2)})</p>
            <h3>All Confidences:</h3>
            <ul>
              {Object.entries(prediction.all_confidences).map(([emotion, confidence]) => (
                <li key={emotion}>
                  {emotion}: {confidence.toFixed(2)}
                </li>
              ))}
            </ul>
          </div>
        )}
      </header>
    </div>
  );
}

export default App
