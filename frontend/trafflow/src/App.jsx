import './App.css';
import { useState } from 'react';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import Header from './components/Header';
import UploadForm from './components/UploadForm';
import VideoProcessor from './components/VideoProcessor';
import EnhancedResults from './components/EnhancedResults';
import LandingPage from './components/LandingPage';
import LoginPage from './pages/LoginPage';
import SignupPage from './pages/SignupPage';
import HistoryPage from './pages/HistoryPage';
import { Container, Spinner } from 'react-bootstrap';
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate
} from 'react-router-dom';

const AppContent = () => {
  const { user, loading } = useAuth();
  const [currentView, setCurrentView] = useState('upload'); // 'upload', 'processing', 'results'
  const [uploadData, setUploadData] = useState(null);
  const [processingComplete, setProcessingComplete] = useState(false);
  const [results, setResults] = useState(null);

  if (loading) {
    return (
      <div className="d-flex justify-content-center align-items-center min-vh-100">
        <Spinner animation="border" role="status">
          <span className="visually-hidden">Loading...</span>
        </Spinner>
      </div>
    );
  }

  if (!user) {
    return (
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/signup" element={<SignupPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    );
  }

  const handleHistoryClick = () => {
    // Navigate to history page
    setCurrentView('history');
  };

  const handleUploadComplete = (result) => {
    setUploadData(result);
    if (result.type === 'video') {
      setCurrentView('processing');
    } else {
      setCurrentView('results');
    }
  };

  const handleProcessingComplete = (finalResults) => {
    setResults(finalResults);
    setProcessingComplete(true);
    setCurrentView('results');
  };

  const handleNewUpload = () => {
    setCurrentView('upload');
    setUploadData(null);
    setProcessingComplete(false);
    setResults(null);
  };

  return (
    <div className="min-vh-100">
      <Header
        onNewUpload={handleNewUpload}
        showNewUpload={currentView !== 'upload'}
        onHistoryClick={handleHistoryClick}
      />

      <main className="py-4">
        {currentView === 'upload' && (
          <UploadForm onUploadComplete={handleUploadComplete} />
        )}

        {currentView === 'processing' && uploadData && (
          <VideoProcessor
            jobData={uploadData}
            onProcessingComplete={handleProcessingComplete}
            onNewUpload={handleNewUpload}
          />
        )}

        {currentView === 'results' && results && (
          <EnhancedResults
            results={results}
            type={uploadData?.type}
            processingComplete={processingComplete}
            onNewUpload={handleNewUpload}
          />
        )}

        {currentView === 'history' && (
          <HistoryPage />
        )}
      </main>
    </div>
  );
};

const App = () => {
  return (
    <AuthProvider>
      <Router>
        <AppContent />
      </Router>
    </AuthProvider>
  );
};

export default App;
