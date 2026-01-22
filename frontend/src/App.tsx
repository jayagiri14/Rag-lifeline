import { useState, useEffect, useRef } from 'react';
import { Send, Loader2, Heart, AlertCircle, RefreshCw, Info, Upload, Stethoscope } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { api, QueryResponse, HistoryInsightResponse, PrescriptionUploadResponse } from './api';

function App() {
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [docsLoaded, setDocsLoaded] = useState<number>(0);
  const [backendStatus, setBackendStatus] = useState<'checking' | 'online' | 'offline'>('checking');

  const [patientId, setPatientId] = useState('demo-patient');
  const [rxFile, setRxFile] = useState<File | null>(null);
  const [uploadState, setUploadState] = useState<{ status: 'idle' | 'uploading'; message: string | null }>({ status: 'idle', message: null });

  const [historySymptoms, setHistorySymptoms] = useState('');
  const [historyInsight, setHistoryInsight] = useState<HistoryInsightResponse | null>(null);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState<string | null>(null);

  // Recording state + refs (added)
  const [recording, setRecording] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  useEffect(() => {
    checkBackendHealth();
  }, []);

  const checkBackendHealth = async () => {
    try {
      const health = await api.healthCheck();
      setBackendStatus('online');
      setDocsLoaded(health.documents_loaded);
    } catch {
      setBackendStatus('offline');
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || loading) return;

    setLoading(true);
    setError(null);

    try {
      const result = await api.query(query);
      setResponse(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to get response. Make sure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!patientId.trim() || !rxFile || uploadState.status === 'uploading') return;
    setUploadState({ status: 'uploading', message: null });
    try {
      const res: PrescriptionUploadResponse = await api.uploadPrescription(patientId.trim(), rxFile);
      setUploadState({ status: 'idle', message: `Stored ${res.stored} item(s) via ${res.engine}` });
    } catch (err) {
      setUploadState({ status: 'idle', message: err instanceof Error ? err.message : 'Upload failed' });
    }
  };

  const handleHistoryInsight = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!patientId.trim() || !historySymptoms.trim() || historyLoading) return;
    setHistoryLoading(true);
    setHistoryError(null);
    try {
      const res = await api.historyInsight(patientId.trim(), historySymptoms.trim(), 6);
      setHistoryInsight(res);
    } catch (err) {
      setHistoryError(err instanceof Error ? err.message : 'Failed to get history insight');
    } finally {
      setHistoryLoading(false);
    }
  };

  const exampleQueries = [
    "I have a runny nose, sneezing, and mild fever. What could it be?",
    "I have sudden weakness after travel with a swollen leg why this happen?",
  ];

  // Recording functions (added)
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;
      audioChunksRef.current = [];

      recorder.ondataavailable = (e: BlobEvent) => {
        if (e.data && e.data.size > 0) audioChunksRef.current.push(e.data);
      };

      recorder.onstop = async () => {
        try {
          const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });

          const res = await api.uploadAudio(patientId.trim(), audioBlob);

          if ((res as any).transcript) {
            setQuery((res as any).transcript);
          }

        } catch (err) {
          console.error('Audio upload failed', err);
        }
      };

      recorder.start();
      setRecording(true);
    } catch (err) {
      console.error('Could not start recording', err);
      setError(err instanceof Error ? err.message : 'Failed to access microphone');
    }
  };

  const stopRecording = () => {
    try {
      mediaRecorderRef.current?.stop();
      mediaRecorderRef.current = null;
      setRecording(false);
    } catch (err) {
      console.error('Error stopping recorder', err);
      setRecording(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-blue-100">
        <div className="max-w-4xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-blue-600 p-2 rounded-lg">
              <Heart className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-800">Medical RAG Assistant</h1>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className={`px-3 py-1 rounded-full text-sm font-medium ${
              backendStatus === 'online' 
                ? 'bg-green-100 text-green-700' 
                : backendStatus === 'offline' 
                ? 'bg-red-100 text-red-700'
                : 'bg-yellow-100 text-yellow-700'
            }`}>
              {backendStatus === 'online' ? `Backend Online` : 
               backendStatus === 'offline' ? '✗ Backend offline' : 
               '⋯ Checking...'}
            </span>
            <button
              onClick={checkBackendHealth}
              className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
              title="Refresh status"
            >
              <RefreshCw className="w-4 h-4 text-gray-500" />
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-4 py-8">
        {/* Disclaimer */}
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 mb-6 flex gap-3">
          <AlertCircle className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
          <div className="text-sm text-amber-800">
            <strong>Medical Disclaimer:</strong> This AI assistant provides general health information only. 
            It is not a substitute for professional medical advice, diagnosis, or treatment. 
            Always consult a qualified healthcare provider for medical concerns.
          </div>
        </div>

        {/* Query Form */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
          <form onSubmit={handleSubmit}>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Describe your symptoms or ask a health question
            </label>
            <div className="flex gap-3">
              <textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="e.g., I have a headache, fatigue, and muscle aches..."
                className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none"
                rows={3}
                disabled={loading || backendStatus === 'offline'}
              />
            </div>
            <div className="flex justify-between items-center mt-4">
              <div className="flex flex-wrap gap-2">
                {exampleQueries.slice(0, 2).map((eq, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => setQuery(eq)}
                    className="text-xs px-3 py-1.5 bg-gray-100 hover:bg-gray-200 text-gray-600 rounded-full transition-colors"
                  >
                    {eq.length > 40 ? eq.substring(0, 40) + '...' : eq}
                  </button>
                ))}
              </div>

              <div className="flex gap-2">
                {/* Record button added */}
                <button
                  type="button"
                  onClick={recording ? stopRecording : startRecording}
                  className={`px-4 py-2 rounded-lg text-white ${
                    recording ? 'bg-red-600' : 'bg-green-600'
                  }`}
                >
                  {recording ? 'Stop' : 'Record'}
                </button>

                <button
                  type="submit"
                  disabled={loading || !query.trim() || backendStatus === 'offline'}
                  className="px-6 py-2.5 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center gap-2 font-medium transition-colors"
                >
                  {loading ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Send className="w-4 h-4" />
                      Ask
                    </>
                  )}
                </button>
              </div>
            </div>
          </form>
        </div>

        {/* History-Based Insight Section */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-6 border border-blue-100">
          <div className="flex items-center gap-2 mb-4">
            <Stethoscope className="w-5 h-5 text-blue-600" />
            <h2 className="text-lg font-semibold text-gray-800">History-Based Medical Insight</h2>
          </div>

          <div className="grid md:grid-cols-2 gap-4">
            {/* Prescription Upload */}
            <form onSubmit={handleUpload} className="p-4 border rounded-lg bg-gray-50 space-y-3">
              <div className="flex items-center gap-2 text-sm font-medium text-gray-700">
                <Upload className="w-4 h-4" /> Upload prescription image
              </div>
              <div className="space-y-2">
                <label className="block text-xs text-gray-600">Patient ID</label>
                <input
                  value={patientId}
                  onChange={(e) => setPatientId(e.target.value)}
                  className="w-full px-3 py-2 border rounded-lg text-sm focus:ring-2 focus:ring-blue-500"
                  placeholder="e.g., patient-123"
                />
              </div>
              <div className="space-y-2">
                <label className="block text-xs text-gray-600">Prescription image</label>
                <input
                  type="file"
                  accept="image/*"
                  onChange={(e) => setRxFile(e.target.files?.[0] || null)}
                  className="w-full text-sm"
                />
              </div>
              <button
                type="submit"
                disabled={!rxFile || !patientId.trim() || uploadState.status === 'uploading'}
                className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 flex items-center justify-center gap-2 text-sm"
              >
                {uploadState.status === 'uploading' ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
                {uploadState.status === 'uploading' ? 'Uploading...' : 'Upload & Store'}
              </button>
              {uploadState.message && (
                <p className="text-xs text-gray-600">{uploadState.message}</p>
              )}
            </form>

            {/* History Insight */}
            <form onSubmit={handleHistoryInsight} className="p-4 border rounded-lg bg-gray-50 space-y-3">
              <div className="flex items-center gap-2 text-sm font-medium text-gray-700">
                <Info className="w-4 h-4 text-blue-600" /> Generate history-based insight
              </div>
              <div className="space-y-2">
                <label className="block text-xs text-gray-600">Patient ID</label>
                <input
                  value={patientId}
                  onChange={(e) => setPatientId(e.target.value)}
                  className="w-full px-3 py-2 border rounded-lg text-sm focus:ring-2 focus:ring-blue-500"
                  placeholder="e.g., patient-123"
                />
              </div>
              <div className="space-y-2">
                <label className="block text-xs text-gray-600">Symptoms</label>
                <textarea
                  value={historySymptoms}
                  onChange={(e) => setHistorySymptoms(e.target.value)}
                  rows={3}
                  className="w-full px-3 py-2 border rounded-lg text-sm focus:ring-2 focus:ring-blue-500"
                  placeholder="e.g., fever, fatigue, slow wound healing"
                />
              </div>
              <button
                type="submit"
                disabled={!patientId.trim() || !historySymptoms.trim() || historyLoading}
                className="w-full px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:bg-gray-400 flex items-center justify-center gap-2 text-sm"
              >
                {historyLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Stethoscope className="w-4 h-4" />}
                {historyLoading ? 'Generating...' : 'Get History Insight'}
              </button>
              {historyError && <p className="text-xs text-red-600">{historyError}</p>}
            </form>
          </div>

          {historyInsight && (
            <div className="mt-5 border-t pt-4">
              <div className="flex items-center gap-2 mb-2">
                <Info className="w-4 h-4 text-indigo-600" />
                <span className="text-sm font-semibold text-gray-800">History-Based Insight</span>
              </div>
              <div className="prose prose-indigo max-w-none text-gray-700 mb-3">
                <ReactMarkdown>{historyInsight.insight}</ReactMarkdown>
              </div>
              <p className="text-xs text-gray-500 mb-3">{historyInsight.disclaimer}</p>
              {historyInsight.history_used?.length > 0 && (
                <div className="space-y-2">
                  <h4 className="text-xs font-semibold text-gray-600">History used ({historyInsight.history_used.length})</h4>
                  {historyInsight.history_used.map((h, idx) => (
                    <div key={idx} className="border rounded-lg p-3 bg-white">
                      <div className="flex justify-between text-xs text-gray-500 mb-1">
                        <span>{h.date || 'date unknown'}</span>
                        <span className={`px-2 py-0.5 rounded-full text-[11px] ${h.is_chronic ? 'bg-red-100 text-red-700' : 'bg-gray-100 text-gray-600'}`}>
                          {h.is_chronic ? 'Chronic' : h.type || 'History'}
                        </span>
                      </div>
                      <p className="text-sm text-gray-700 mb-1">{h.summary}</p>
                      {h.raw_text && <p className="text-xs text-gray-500 line-clamp-2">{h.raw_text}</p>}
                      <div className="text-[11px] text-gray-500">Weight: {h.score.toFixed(2)}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Error Message */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6 flex gap-3">
            <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0" />
            <div className="text-red-700">{error}</div>
          </div>
        )}

        {/* Response */}
        {response && (
          <div className="bg-white rounded-xl shadow-lg overflow-hidden">
            {/* Response Content */}
            <div className="p-6">
              <h2 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
                <Info className="w-5 h-5 text-blue-600" />
                Medical Information
              </h2>
              <div className="prose prose-blue max-w-none markdown-content text-gray-700">
                <ReactMarkdown>{response.response}</ReactMarkdown>
              </div>
            </div>
          </div>
        )}

        {/* Empty State */}
        {!response && !loading && !error && (
          <div className="text-center py-12">
            <div className="bg-blue-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
              <Heart className="w-8 h-8 text-blue-600" />
            </div>
            <h3 className="text-lg font-medium text-gray-700 mb-2">
              Ask about your symptoms
            </h3>
            <p className="text-gray-500 mb-6">
              Describe what you're experiencing and get helpful medical information
            </p>
            <div className="flex flex-wrap justify-center gap-2">
              
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="text-center py-6 text-sm text-gray-500">
        <p>Medical RAG System • Using Qdrant</p>
      </footer>
    </div>
  );
}

export default App;