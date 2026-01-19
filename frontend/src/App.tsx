import { useState, useEffect } from 'react';
import { Send, Loader2, Heart, AlertCircle, RefreshCw, Info } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { api, QueryResponse } from './api';

function App() {
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [docsLoaded, setDocsLoaded] = useState<number>(0);
  const [backendStatus, setBackendStatus] = useState<'checking' | 'online' | 'offline'>('checking');

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

  const exampleQueries = [
    "I have a runny nose, sneezing, and mild fever. What could it be?",
    "What are the symptoms of a migraine?",
    "I've been having trouble sleeping for weeks. What should I do?",
    "What causes high blood pressure?",
  ];

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
              <p className="text-sm text-gray-500">Powered by DeepSeek R1</p>
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
              {backendStatus === 'online' ? `✓ ${docsLoaded} docs loaded` : 
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
          </form>
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

            {/* Sources */}
            {response.sources && response.sources.length > 0 && (
              <div className="border-t border-gray-100 bg-gray-50 p-6">
                <h3 className="text-sm font-semibold text-gray-600 mb-3">
                  Related Conditions ({response.sources.length})
                </h3>
                <div className="grid gap-3">
                  {response.sources.map((source, idx) => (
                    <div key={idx} className="bg-white rounded-lg p-3 border border-gray-200">
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-medium text-blue-700">{source.condition}</span>
                        <span className="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full">
                          {(source.relevance_score * 100).toFixed(0)}% match
                        </span>
                      </div>
                      <p className="text-sm text-gray-600 line-clamp-2">{source.content}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Model Info */}
            <div className="border-t border-gray-100 px-6 py-3 bg-gray-50 flex items-center justify-between text-xs text-gray-500">
              <span>Model: {response.model}</span>
              {response.usage && (
                <span>Tokens: {response.usage.total_tokens || 'N/A'}</span>
              )}
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
              {exampleQueries.map((eq, idx) => (
                <button
                  key={idx}
                  onClick={() => setQuery(eq)}
                  className="text-sm px-4 py-2 bg-white border border-gray-200 hover:border-blue-300 hover:bg-blue-50 text-gray-700 rounded-lg transition-colors"
                >
                  {eq}
                </button>
              ))}
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="text-center py-6 text-sm text-gray-500">
        <p>Medical RAG System • Using Qdrant + LangChain + DeepSeek R1</p>
      </footer>
    </div>
  );
}

export default App;
