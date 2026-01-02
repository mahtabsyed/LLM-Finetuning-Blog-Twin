/**
 * Comparison View Component
 *
 * Displays side-by-side comparison of base and fine-tuned models.
 * Allows users to see the difference in writing style and quality.
 */

import React from 'react';

function ComparisonView({ comparison }) {
  const { base_model, finetuned_model } = comparison;

  return (
    <div className="space-y-6">
      {/* Comparison header */}
      <div className="bg-white rounded-lg shadow-sm px-6 py-4 border border-gray-200">
        <h3 className="font-semibold text-gray-800 mb-2">Model Comparison</h3>
        <p className="text-sm text-gray-600">
          Comparing responses to: <span className="italic">"{comparison.prompt}"</span>
        </p>
      </div>

      {/* Side-by-side comparison */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Base model response */}
        <ModelResponse
          title="Base Model"
          subtitle={base_model.model_name}
          response={base_model}
          color="blue"
        />

        {/* Fine-tuned model response */}
        <ModelResponse
          title="Fine-tuned Model (Your Twin)"
          subtitle={finetuned_model.model_name}
          response={finetuned_model}
          color="orange"
        />
      </div>

      {/* Comparison insights */}
      {base_model.success && finetuned_model.success && (
        <div className="bg-gradient-to-r from-gray-50 to-gray-100 rounded-lg p-6 border border-gray-200">
          <h4 className="font-medium text-gray-800 mb-3">Quick Stats</h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <p className="text-gray-500">Base tokens</p>
              <p className="font-semibold text-gray-800">
                {base_model.tokens_generated || 'N/A'}
              </p>
            </div>
            <div>
              <p className="text-gray-500">Fine-tuned tokens</p>
              <p className="font-semibold text-gray-800">
                {finetuned_model.tokens_generated || 'N/A'}
              </p>
            </div>
            <div>
              <p className="text-gray-500">Base time</p>
              <p className="font-semibold text-gray-800">
                {base_model.generation_time ? `${base_model.generation_time.toFixed(2)}s` : 'N/A'}
              </p>
            </div>
            <div>
              <p className="text-gray-500">Fine-tuned time</p>
              <p className="font-semibold text-gray-800">
                {finetuned_model.generation_time ? `${finetuned_model.generation_time.toFixed(2)}s` : 'N/A'}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Individual model response card
 */
function ModelResponse({ title, subtitle, response, color }) {
  const colorClasses = {
    blue: {
      header: 'from-blue-50 to-blue-100 border-blue-200',
      badge: 'bg-blue-100 text-blue-800',
    },
    orange: {
      header: 'from-orange-50 to-orange-100 border-orange-200',
      badge: 'bg-orange-100 text-orange-800',
    },
  };

  const colors = colorClasses[color];

  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden">
      {/* Header */}
      <div className={`bg-gradient-to-r ${colors.header} px-4 py-3 border-b`}>
        <div className="flex items-center justify-between">
          <div>
            <h4 className="font-semibold text-gray-800">{title}</h4>
            <p className="text-xs text-gray-600">{subtitle}</p>
          </div>
          {response.success && (
            <span className={`${colors.badge} text-xs px-2 py-1 rounded-full`}>
              ✓ Success
            </span>
          )}
        </div>
      </div>

      {/* Content */}
      <div className="p-4">
        {response.success ? (
          <p className="text-gray-800 text-sm whitespace-pre-wrap leading-relaxed">
            {response.text}
          </p>
        ) : (
          <div className="text-red-600 text-sm">
            <p className="font-medium mb-1">Generation failed</p>
            <p>This model might not be available.</p>
          </div>
        )}
      </div>

      {/* Footer with stats */}
      {response.success && (
        <div className="px-4 py-2 bg-gray-50 border-t border-gray-100 flex justify-between text-xs text-gray-500">
          <span>⚡ {response.generation_time?.toFixed(2)}s</span>
          <span>📝 {response.tokens_generated} tokens</span>
        </div>
      )}
    </div>
  );
}

export default ComparisonView;
