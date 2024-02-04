/// <reference types="vitest" />

import { defineConfig } from 'vitest/config';

export default defineConfig({
  // Configuration specific to the second group of tests
  test: {
    // Test-specific configurations
    browser: {
      enabled: true,
      name: 'chrome'
    }
  }
});
