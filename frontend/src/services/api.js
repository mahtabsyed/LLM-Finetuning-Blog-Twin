/**
 * API Service
 *
 * This module handles all communication with the FastAPI backend.
 * It provides functions for:
 * - Generating text from models
 * - Getting model information
 * - Comparing model outputs
 * - Health checks
 */

import axios from 'axios';

// Base URL for API - can be configured via environment variable
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000, // 2 minutes timeout for long generations
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Generate text from a model
 *
 * @param {Object} params - Generation parameters
 * @param {string} params.prompt - The input prompt
 * @param {string} params.model - Model type: 'base' or 'finetuned'
 * @param {number} params.maxTokens - Maximum tokens to generate
 * @param {number} params.temperature - Sampling temperature
 * @returns {Promise<Object>} Generated response
 */
export const generateText = async ({
  prompt,
  model = 'finetuned',
  maxTokens = 500,
  temperature = 0.7,
  topP = 0.9,
}) => {
  try {
    const response = await apiClient.post('/api/generate', {
      prompt,
      model,
      max_tokens: maxTokens,
      temperature,
      top_p: topP,
    });

    return {
      success: true,
      data: response.data,
    };
  } catch (error) {
    console.error('Error generating text:', error);
    return {
      success: false,
      error: error.response?.data?.detail || error.message || 'Generation failed',
    };
  }
};

/**
 * Get information about available models
 *
 * @returns {Promise<Object>} List of models with their status
 */
export const getModels = async () => {
  try {
    const response = await apiClient.get('/api/models');

    return {
      success: true,
      data: response.data,
    };
  } catch (error) {
    console.error('Error fetching models:', error);
    return {
      success: false,
      error: error.response?.data?.detail || error.message || 'Failed to fetch models',
    };
  }
};

/**
 * Compare base and fine-tuned models side-by-side
 *
 * @param {Object} params - Comparison parameters
 * @param {string} params.prompt - The input prompt
 * @param {number} params.maxTokens - Maximum tokens to generate
 * @param {number} params.temperature - Sampling temperature
 * @returns {Promise<Object>} Comparison results from both models
 */
export const compareModels = async ({
  prompt,
  maxTokens = 500,
  temperature = 0.7,
}) => {
  try {
    const response = await apiClient.post('/api/compare', {
      prompt,
      max_tokens: maxTokens,
      temperature,
    });

    return {
      success: true,
      data: response.data,
    };
  } catch (error) {
    console.error('Error comparing models:', error);
    return {
      success: false,
      error: error.response?.data?.detail || error.message || 'Comparison failed',
    };
  }
};

/**
 * Check API and Ollama health status
 *
 * @returns {Promise<Object>} Health status information
 */
export const checkHealth = async () => {
  try {
    const response = await apiClient.get('/health');

    return {
      success: true,
      data: response.data,
    };
  } catch (error) {
    console.error('Error checking health:', error);
    return {
      success: false,
      error: error.message || 'Health check failed',
    };
  }
};

export default {
  generateText,
  getModels,
  compareModels,
  checkHealth,
};
