import { useState } from 'react';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:5001/api';

function App() {
  const [file, setFile] = useState(null);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
    setError(null);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!file) {
      setError('Please select a file');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(`${API_BASE_URL}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.data.type === 'video') {
        // Redirect to video processing page
        window.location.href = `/jobs/${response.data.job_id}`;
      } else {
        setResults(response.data.results);
      }
    } catch (error) {
      setError(error.response?.data?.error || 'An error occurred during upload');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>License Plate Detection</h1>
      
      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      <form onSubmit={handleSubmit}>
        <input
          type="file"
          onChange={handleFileChange}
          accept=".jpg,.jpeg,.png,.mp4,.avi,.mov"
        />
        <button type="submit" disabled={loading}>
          {loading ? 'Processing...' : 'Upload'}
        </button>
      </form>

      {results && (
        <div className="results">
          <h2>Results</h2>
          {results.map((result, index) => (
            <div key={index} className="result-item">
              <img src={result.image_data} alt={`Detection ${index + 1}`} />
              <p>Plate: {result.plate_text}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default App;
