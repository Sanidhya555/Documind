/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        brand: {
          50: '#f0f7ff',
          100: '#e0effe',
          200: '#b9dffe',
          300: '#7cc3fd',
          400: '#36a3f9',
          500: '#0c85eb',
          600: '#0267c8',
          700: '#0352a2',
          800: '#074685',
          900: '#0c3b6e',
          950: '#082548',
        }
      }
    },
  },
  plugins: [],
}
