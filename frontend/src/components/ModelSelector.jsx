/**
 * Model Selector Component
 *
 * Allows users to choose between base and fine-tuned models.
 * Displays model availability and descriptions.
 */

import React from 'react';

function ModelSelector({ models, selectedModel, onModelChange }) {
  // Find the currently selected model details
  const currentModel = models.find(m => m.type === selectedModel);

  return (
    <div className="bg-white rounded-lg shadow-sm p-4 border border-gray-200">
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Select Model
          </label>
          <div className="flex space-x-2">
            {models.map((model) => (
              <button
                key={model.type}
                onClick={() => model.available && onModelChange(model.type)}
                disabled={!model.available}
                className={`
                  px-4 py-2 rounded-lg font-medium transition-all
                  ${selectedModel === model.type
                    ? 'bg-orange-500 text-white shadow-md'
                    : model.available
                      ? 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                      : 'bg-gray-50 text-gray-400 cursor-not-allowed'
                  }
                `}
              >
                {model.type === 'base' ? '🌐 Base Model' : '✨ Fine-tuned'}
                {!model.available && ' (unavailable)'}
              </button>
            ))}
          </div>
        </div>

        {/* Current model info */}
        {currentModel && (
          <div className="ml-6 text-right">
            <p className="text-sm font-medium text-gray-700">
              {currentModel.name}
            </p>
            <p className="text-xs text-gray-500 max-w-xs">
              {currentModel.description}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default ModelSelector;
