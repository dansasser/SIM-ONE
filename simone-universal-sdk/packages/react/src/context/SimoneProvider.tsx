import React, { createContext, useContext, useMemo } from 'react';
import { SimoneClient, SimoneClientConfig } from '@simone/core';

// Define the shape of the context value
interface SimoneContextType {
  client: SimoneClient | null;
}

// Create the context with a default value of null
const SimoneContext = createContext<SimoneContextType>({ client: null });

/**
 * A custom hook to easily access the SimoneClient instance from the context.
 * Throws an error if used outside of a SimoneProvider.
 * @returns The SimoneClient instance.
 */
export const useSimone = (): SimoneClient => {
  const context = useContext(SimoneContext);
  if (!context || !context.client) {
    throw new Error('useSimone must be used within a SimoneProvider');
  }
  return context.client;
};

// Define the props for the provider component
interface SimoneProviderProps {
  config: SimoneClientConfig;
  children: React.ReactNode;
}

/**
 * A React Provider component that initializes and provides a SimoneClient instance
 * to its children.
 */
export const SimoneProvider: React.FC<SimoneProviderProps> = ({ config, children }) => {
  // useMemo ensures the client is only created once as long as the config doesn't change.
  const client = useMemo(() => {
    if (!config) {
      return null;
    }
    return new SimoneClient(config);
  }, [config]);

  return (
    <SimoneContext.Provider value={{ client }}>
      {children}
    </SimoneContext.Provider>
  );
};
