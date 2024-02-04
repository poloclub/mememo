module.exports = {
  parser: '@typescript-eslint/parser',
  extends: [
    'eslint:recommended',
    'plugin:@typescript-eslint/recommended',
    'plugin:@typescript-eslint/recommended-requiring-type-checking',
    'prettier'
  ],
  parserOptions: {
    ecmaVersion: 'latest',
    sourceType: 'module',
    project: ['./typescript.eslintrc.json'],
    extraFileExtensions: ['.cjs']
  },
  plugins: ['@typescript-eslint', 'prettier'],
  env: {
    es6: true,
    browser: true
  },
  ignorePatterns: ['node_modules'],
  rules: {
    indent: ['error', 2, { SwitchCase: 1 }],
    'linebreak-style': ['error', 'unix'],
    quotes: ['error', 'single'],
    'prefer-const': ['error'],
    semi: ['error', 'always'],
    'max-len': [
      'warn',
      {
        code: 80
      }
    ],
    'prettier/prettier': 2,
    '@typescript-eslint/ban-ts-comment': 'off',
    '@typescript-eslint/restrict-template-expressions': 'off',
    '@typescript-eslint/no-non-null-assertion': 'off',
    '@typescript-eslint/no-empty-function': 'off',
    '@typescript-eslint/no-unused-vars': ['warn'],
    'no-self-assign': 'off'
  }
};
