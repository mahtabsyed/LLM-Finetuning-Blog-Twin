/**
 * Header Component
 *
 * Displays the application header with:
 * - Logo/branding
 * - Health status indicator
 */

import React from 'react';

function Header({ apiHealth }) {
  return (
    <header className="bg-white border-b border-gray-200">
      <div className="container mx-auto px-4 py-4">
        <div className="flex justify-between items-center">
          {/* Logo and title */}
          <div className="flex items-center space-x-3">
            <div className="text-3xl">✍️</div>
            <div>
              <h1 className="text-2xl font-bold text-gray-800">
                Blogging Twin
              </h1>
              <p className="text-sm text-gray-500">
                Your AI writing assistant
              </p>
            </div>
          </div>

          {/* Health status */}
          <div className="flex items-center space-x-2">
            {apiHealth && apiHealth.ollama_connected ? (
              <>
                <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse"></div>
                <span className="text-sm text-gray-600">
                  {apiHealth.models_available} {apiHealth.models_available === 1 ? 'model' : 'models'} ready
                </span>
              </>
            ) : (
              <>
                <div className="h-2 w-2 rounded-full bg-red-500"></div>
                <span className="text-sm text-gray-600">Disconnected</span>
              </>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}

export default Header;
