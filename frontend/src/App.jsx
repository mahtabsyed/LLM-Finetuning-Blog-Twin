/**
 * Main App Component
 *
 * This is the root component of the blogging twin interface.
 * It manages the overall state and layout, coordinating between components.
 *
 * Design inspired by claude.ai with:
 * - Clean, minimal greeting interface
 * - Model selector showing current model
 * - Input prompt area
 * - Response display
 * - Comparison mode option
 */

import { useState, useEffect } from 'react';
import { getModels, checkHealth } from './services/api';
import ChatInterface from './components/ChatInterface';
import ModelSelector from './components/ModelSelector';
import Header from './components/Header';

function App() {
  // State management
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('finetuned');
  const [apiHealth, setApiHealth] = useState(null);
  const [loading, setLoading] = useState(true);

  // Load models and check health on component mount
  useEffect(() => {
    loadInitialData();
  }, []);

  /**
   * Load models and check API health
   */
  const loadInitialData = async () => {
    setLoading(true);

    // Check API health
    const healthResult = await checkHealth();
    setApiHealth(healthResult.data);

    // Get available models
    const modelsResult = await getModels();
    if (modelsResult.success) {
      setModels(modelsResult.data.models);

      // Default to fine-tuned if available, otherwise base
      const finetunedAvailable = modelsResult.data.models.find(
        m => m.type === 'finetuned' && m.available
      );
      setSelectedModel(finetunedAvailable ? 'finetuned' : 'base');
    }

    setLoading(false);
  };

  /**
   * Handle model selection change
   */
  const handleModelChange = (modelType) => {
    setSelectedModel(modelType);
  };

  // Show loading state while initializing
  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-orange-500 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading your blogging twin...</p>
        </div>
      </div>
    );
  }

  // Show error state if API is unhealthy
  if (apiHealth && !apiHealth.ollama_connected) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100 flex items-center justify-center">
        <div className="max-w-md mx-auto bg-white rounded-lg shadow-lg p-8">
          <div className="text-center">
            <div className="text-red-500 text-5xl mb-4">⚠️</div>
            <h2 className="text-2xl font-bold text-gray-800 mb-4">
              Cannot Connect to Ollama
            </h2>
            <p className="text-gray-600 mb-6">
              The backend cannot connect to your local Ollama server.
              Please ensure Ollama is running.
            </p>
            <div className="bg-gray-100 rounded p-4 text-left">
              <code className="text-sm text-gray-800">
                $ ollama serve
              </code>
            </div>
            <button
              onClick={loadInitialData}
              className="mt-6 bg-orange-500 hover:bg-orange-600 text-white font-medium py-2 px-6 rounded-lg transition-colors"
            >
              Retry Connection
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100 flex flex-col">
      {/* Header with branding */}
      <Header apiHealth={apiHealth} />

      {/* Main content area */}
      <main className="flex-1 container mx-auto px-4 py-4">
        {/* Model selector */}
        <div className="max-w-4xl mx-auto mb-4">
          <ModelSelector
            models={models}
            selectedModel={selectedModel}
            onModelChange={handleModelChange}
          />
        </div>

        {/* Chat interface */}
        <ChatInterface selectedModel={selectedModel} />
      </main>

      {/* Footer - Always at bottom */}
      <footer className="text-center py-4 text-gray-500 text-sm bg-white border-t mt-auto">
        <p>LLM Blogging Twin - Fine-tuned on your personal blog posts</p>
        <p className="mt-1">
          Powered by{' '}
          <a
            href="https://ollama.ai"
            target="_blank"
            rel="noopener noreferrer"
            className="text-orange-500 hover:underline"
          >
            Ollama
          </a>
          {' '}and{' '}
          <a
            href="https://github.com/unslothai/unsloth"
            target="_blank"
            rel="noopener noreferrer"
            className="text-orange-500 hover:underline"
          >
            Unsloth
          </a>
        </p>
      </footer>
    </div>
  );
}

export default App;
