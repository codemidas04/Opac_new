import React from "react";
import CreditForm from "./components/CreditForm";
import "./App.css";

function App() {
  return (
    <div className="App">
      <header>
        <h1>OpacGuard — Credit Risk Demo</h1>
      </header>
      <main>
        <CreditForm />
      </main>
    </div>
  );
}

export default App;