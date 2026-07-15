/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#16181d",
        paper: "#f7f4ee",
        moss: "#267365",
        saffron: "#d89a2b",
        berry: "#9b3a5b"
      }
    }
  },
  plugins: []
};
