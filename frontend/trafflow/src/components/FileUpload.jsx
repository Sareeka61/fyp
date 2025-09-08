import React, { useState } from 'react';
import axios from 'axios';
import { API_BASE, WS_BASE } from '../config';

const FileUpload = ({ onUploadSuccess }) => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);
    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(`${API_BASE}/upload`, formData);
      onUploadSuccess(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Upload failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow-md">
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
          <input
            type="file"
            onChange={(e) => setFile(e.target.files[0])}
            className="hidden"
            id="file-upload"
            accept="video/*,image/*"
          />
          <label 
            htmlFor="file-upload"
            className="cursor-pointer text-blue-600 hover:text-blue-800"
          >
            Choose a file
          </label>
          {file && <p className="mt-2 text-gray-600">{file.name}</p>}
        </div>
        
        <button
          type="submit"
          disabled={!file || loading}
          className="w-full bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700 disabled:bg-gray-400"
        >
          {loading ? 'Uploading...' : 'Upload'}
        </button>
        
        {error && (
          <div className="text-red-600 text-sm mt-2">{error}</div>
        )}
      </form>
    </div>
  );
};

export default FileUpload;
