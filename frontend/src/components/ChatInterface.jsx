/**
 * Chat Interface Component
 *
 * Main interface for interacting with the blogging twin.
 * Features:
 * - Greeting message (centered when no conversation)
 * - Conversation history with user/AI messages
 * - Fixed input at bottom during conversation
 * - Suggested prompts
 * - Comparison mode option
 *
 * Design inspired by claude.ai
 */

import { useState, useRef, useEffect } from 'react';
import { generateText, compareModels } from '../services/api';
import PromptInput from './PromptInput';
import ComparisonView from './ComparisonView';

function ChatInterface({ selectedModel }) {
  const [prompt, setPrompt] = useState('');
  const [messages, setMessages] = useState([]); // Conversation history
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showComparison, setShowComparison] = useState(false);
  const messagesEndRef = useRef(null);

  // Scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Suggested prompts to help users get started
  const suggestedPrompts = [
    'Write a blog post about the future of AI',
    'Share your thoughts on remote work',
    'Explain machine learning to beginners',
    'Discuss the importance of mentorship',
  ];

  /**
   * Handle prompt submission
   */
  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!prompt.trim()) return;

    const userMessage = prompt.trim();
    setPrompt(''); // Clear input immediately

    // Add user message to conversation
    setMessages(prev => [...prev, {
      role: 'user',
      content: userMessage,
      timestamp: new Date()
    }]);

    setLoading(true);
    setError(null);

    try {
      if (showComparison) {
        // Compare both models
        const result = await compareModels({
          prompt: userMessage,
          maxTokens: 500,
          temperature: 0.7,
        });

        if (result.success) {
          setMessages(prev => [...prev, {
            role: 'assistant',
            content: result.data,
            isComparison: true,
            timestamp: new Date()
          }]);
        } else {
          setError(result.error);
        }
      } else {
        // Generate from selected model only
        const result = await generateText({
          prompt: userMessage,
          model: selectedModel,
          maxTokens: 500,
          temperature: 0.7,
        });

        if (result.success) {
          setMessages(prev => [...prev, {
            role: 'assistant',
            content: result.data,
            timestamp: new Date()
          }]);
        } else {
          setError(result.error);
        }
      }
    } catch (err) {
      setError('An unexpected error occurred. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  /**
   * Handle selecting a suggested prompt
   */
  const handleSuggestedPrompt = (suggestedPrompt) => {
    setPrompt(suggestedPrompt);
  };

  /**
   * Clear conversation and start fresh
   */
  const handleClear = () => {
    setPrompt('');
    setMessages([]);
    setError(null);
  };

  // Show greeting screen if no messages
  const showGreeting = messages.length === 0 && !loading;
  const hasConversation = messages.length > 0 || loading;

  return (
    <div className="flex flex-col" style={{ height: 'calc(90vh - 280px)' }}>
      {/* Greeting - Centered when no conversation */}
      {showGreeting && (
        <div className="flex-1 flex flex-col items-center justify-center max-w-4xl mx-auto w-full px-4">
          <div className="text-center mb-12">
            <h2 className="text-4xl font-bold text-gray-800 mb-4 flex items-center justify-center">
              <span className="mr-3">☀️</span>
              <span>Good {getTimeOfDay()}</span>
            </h2>
            <p className="text-lg text-gray-600">
              What would you like to write about today?
            </p>
          </div>

          {/* Centered Prompt input for greeting */}
          <div className="w-full max-w-3xl bg-white rounded-2xl shadow-lg p-6 mb-8">
            <form onSubmit={handleSubmit}>
              <PromptInput
                value={prompt}
                onChange={setPrompt}
                onSubmit={handleSubmit}
                loading={loading}
                disabled={loading}
              />

              {/* Comparison mode toggle */}
              <div className="mt-4 flex items-center justify-between">
                <label className="flex items-center space-x-2 text-sm text-gray-600 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={showComparison}
                    onChange={(e) => setShowComparison(e.target.checked)}
                    className="rounded text-orange-500 focus:ring-orange-500"
                  />
                  <span>Compare base vs. fine-tuned models</span>
                </label>
              </div>
            </form>
          </div>

          {/* Suggested prompts */}
          <div className="w-full max-w-3xl">
            <p className="text-sm text-gray-500 mb-3">Try these prompts:</p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {suggestedPrompts.map((suggested, index) => (
                <button
                  key={index}
                  onClick={() => handleSuggestedPrompt(suggested)}
                  className="text-left p-4 bg-white rounded-lg border border-gray-200 hover:border-orange-300 hover:shadow-md transition-all"
                >
                  <div className="flex items-start">
                    <span className="text-gray-400 mr-3">💡</span>
                    <span className="text-sm text-gray-700">{suggested}</span>
                  </div>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Conversation View */}
      {hasConversation && (
        <>
          {/* Messages Area - Scrollable */}
          <div className="flex-1 overflow-y-auto px-4 pb-4">
            <div className="max-w-4xl mx-auto space-y-6 py-8">
              {messages.map((message, index) => (
                <div key={index}>
                  {message.role === 'user' ? (
                    <UserMessage content={message.content} />
                  ) : message.isComparison ? (
                    <ComparisonView comparison={message.content} />
                  ) : (
                    <AssistantMessage response={message.content} />
                  )}
                </div>
              ))}

              {/* Loading state */}
              {loading && (
                <div className="flex items-start space-x-4">
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-orange-100 flex items-center justify-center">
                    <span className="text-orange-600">✨</span>
                  </div>
                  <div className="bg-white rounded-lg shadow-sm p-6 flex-1">
                    <div className="flex items-center space-x-3">
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-orange-500"></div>
                      <p className="text-gray-600 text-sm">
                        {showComparison ? 'Comparing models...' : 'Generating response...'}
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Error display */}
              {error && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-6">
                  <div className="flex items-start">
                    <span className="text-red-500 text-xl mr-3">⚠️</span>
                    <div>
                      <h3 className="font-medium text-red-800 mb-1">Error</h3>
                      <p className="text-red-700">{error}</p>
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          </div>

          {/* Fixed Input at Bottom */}
          <div className="border-t bg-white">
            <div className="max-w-4xl mx-auto px-4 py-4">
              <form onSubmit={handleSubmit}>
                <PromptInput
                  value={prompt}
                  onChange={setPrompt}
                  onSubmit={handleSubmit}
                  loading={loading}
                  disabled={loading}
                />

                {/* Options row */}
                <div className="mt-3 flex items-center justify-between">
                  <label className="flex items-center space-x-2 text-sm text-gray-600 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={showComparison}
                      onChange={(e) => setShowComparison(e.target.checked)}
                      className="rounded text-orange-500 focus:ring-orange-500"
                    />
                    <span>Compare models</span>
                  </label>

                  <button
                    type="button"
                    onClick={handleClear}
                    className="text-sm text-gray-500 hover:text-gray-700"
                  >
                    Clear & start new
                  </button>
                </div>
              </form>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

/**
 * User Message Component
 */
function UserMessage({ content }) {
  return (
    <div className="flex justify-end">
      <div className="bg-gray-100 rounded-2xl px-4 py-3 max-w-2xl">
        <p className="text-gray-800">{content}</p>
      </div>
    </div>
  );
}

/**
 * Assistant Message Component
 */
function AssistantMessage({ response }) {
  return (
    <div className="flex items-start space-x-4">
      {/* Avatar */}
      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-orange-100 flex items-center justify-center">
        <span className="text-orange-600">✨</span>
      </div>

      {/* Message content */}
      <div className="flex-1 bg-white rounded-lg border border-gray-200 overflow-hidden">
        {/* Header with model info */}
        <div className="bg-gray-50 px-4 py-2 border-b border-gray-200 flex items-center justify-between">
          <div className="text-sm text-gray-600">
            <span className="font-medium">{response.model_used}</span>
          </div>
          <div className="flex items-center space-x-3 text-xs text-gray-500">
            {response.generation_time && (
              <span>⚡ {response.generation_time.toFixed(2)}s</span>
            )}
            {response.tokens_generated && (
              <span>📝 {response.tokens_generated} tokens</span>
            )}
          </div>
        </div>

        {/* Response text */}
        <div className="p-4">
          <p className="text-gray-800 whitespace-pre-wrap leading-relaxed">
            {response.text}
          </p>
        </div>
      </div>
    </div>
  );
}

/**
 * Get time of day for greeting
 */
function getTimeOfDay() {
  const hour = new Date().getHours();
  if (hour < 12) return 'Morning';
  if (hour < 18) return 'Afternoon';
  return 'Evening';
}

export default ChatInterface;
