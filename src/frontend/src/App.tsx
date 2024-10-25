import AppMain from "./AppMain.js";
import Home from "./Home.js";
import { Routes, Route } from "react-router-dom";

function App() {
  return (
    <div>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/main" element={<AppMain />} />
      </Routes>
    </div>
  );
}

export default App;
