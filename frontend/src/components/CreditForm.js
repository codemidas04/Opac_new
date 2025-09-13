import React, { useState } from "react";
import ExplanationChart from "../components/ExplanationChart"; // ✅ import chart

function CreditForm({ onResult }) {
const [formData, setFormData] = useState({
    Duration: "",
    CreditAmount: "",
    Age: "",
    InstallmentRate: "",
    ResidenceDuration: "",
    ExistingCredits: "",
    LiableDependents: "",
    Status: "A11",
    CreditHistory: "A30",
    Purpose: "A40",
    Savings: "A61",
    Employment: "A71",
    PersonalStatusSex: "A91",
    OtherDebtors: "A101",
    Property: "A121",
    OtherInstallmentPlans: "A141",
    Housing: "A151",
    Job: "A171",
    Telephone: "A191",
    ForeignWorker: "A201",
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [explanation, setExplanation] = useState(null); // ✅ new state

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    setExplanation(null);


    console.log("Submitting data:", formData);

    fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formData),
    })
      .then((res) => {
        if (!res.ok) throw new Error("Network response was not ok");
        return res.json();
      })
      .then((data) => {
        console.log("API response:", data);
        setResult(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Error:", err);
        setError(err.message);
        setLoading(false);
      });
  };
   // ✅ new explainability fetch
  const handleExplain = () => {
    setLoading(true);
    setError(null);

    fetch("http://127.0.0.1:5000/explain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formData),
    })
      .then((res) => {
        if (!res.ok) throw new Error("Explainability request failed");
        return res.json();
      })
      .then((data) => {
        setExplanation(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Explain error:", err);
        setError(err.message);
        setLoading(false);
      });
  };
  return (
    <div className="credit-form">
      <h2>Credit Risk Prediction</h2>
      <form onSubmit={handleSubmit}>
        {/* Example field */}

      {/* Duration */}
      <label>Duration (months):</label>
      <input
        type="number"
        name="Duration"
        value={formData.Duration}
        onChange={handleChange}
        required
      />

      {/* Credit Amount */}
      <label>Credit Amount:</label>
      <input
        type="number"
        name="CreditAmount"
        value={formData.CreditAmount}
        onChange={handleChange}
        required
      />

      {/* Age */}
      <label>Age:</label>
      <input
        type="number"
        name="Age"
        value={formData.Age}
        onChange={handleChange}
        required
      />

      {/* Installment Rate */}
      <label>Installment Rate (% of income):</label>
      <input
        type="number"
        name="InstallmentRate"
        value={formData.InstallmentRate}
        onChange={handleChange}
        required
      />

      {/* Residence Duration */}
      <label>Residence Duration (years):</label>
      <input
        type="number"
        name="ResidenceDuration"
        value={formData.ResidenceDuration}
        onChange={handleChange}
        required
      />

      {/* Existing Credits */}
      <label>Existing Credits:</label>
      <input
        type="number"
        name="ExistingCredits"
        value={formData.ExistingCredits}
        onChange={handleChange}
        required
      />

      {/* Dependents */}
      <label>Number of Dependents:</label>
      <input
        type="number"
        name="LiableDependents"
        value={formData.LiableDependents}
        onChange={handleChange}
        required
      />

      {/* Dropdowns */}
      <label>Status of Checking Account:</label>
      <select name="Status" value={formData.Status} onChange={handleChange}>
        <option value="A11">&lt; 0 DM</option>
        <option value="A12">0 ≤ ... &lt; 200 DM</option>
        <option value="A13">≥ 200 DM</option>
        <option value="A14">No checking account</option>
      </select>

      <label>Credit History:</label>
      <select
        name="CreditHistory"
        value={formData.CreditHistory}
        onChange={handleChange}
      >
        <option value="A30">No credits taken / all paid duly</option>
        <option value="A31">All credits at this bank paid duly</option>
        <option value="A32">Existing credits paid duly till now</option>
        <option value="A33">Delay in past payments</option>
        <option value="A34">Critical account / other credits</option>
      </select>

      <label>Purpose:</label>
      <select name="Purpose" value={formData.Purpose} onChange={handleChange}>
        <option value="A40">Car (new)</option>
        <option value="A41">Car (used)</option>
        <option value="A42">Furniture / Equipment</option>
        <option value="A43">Radio / Television</option>
        <option value="A44">Domestic appliances</option>
        <option value="A45">Repairs</option>
        <option value="A46">Education</option>
        <option value="A47">Vacation</option>
        <option value="A48">Retraining</option>
        <option value="A49">Business</option>
        <option value="A410">Others</option>
      </select>

      <label>Savings:</label>
      <select name="Savings" value={formData.Savings} onChange={handleChange}>
        <option value="A61">&lt; 100 DM</option>
        <option value="A62">100 ≤ ... &lt; 500 DM</option>
        <option value="A63">500 ≤ ... &lt; 1000 DM</option>
        <option value="A64">≥ 1000 DM</option>
        <option value="A65">No savings account</option>
      </select>

      <label>Employment:</label>
      <select
        name="Employment"
        value={formData.Employment}
        onChange={handleChange}
      >
        <option value="A71">Unemployed</option>
        <option value="A72">&lt; 1 year</option>
        <option value="A73">1 ≤ ... &lt; 4 years</option>
        <option value="A74">4 ≤ ... &lt; 7 years</option>
        <option value="A75">≥ 7 years</option>
      </select>

      <label>Personal Status & Sex:</label>
      <select
        name="PersonalStatusSex"
        value={formData.PersonalStatusSex}
        onChange={handleChange}
      >
        <option value="A91">Male: divorced/separated</option>
        <option value="A92">Female: divorced/separated/married</option>
        <option value="A93">Male: single</option>
        <option value="A94">Male: married/widowed</option>
        <option value="A95">Female: single</option>
      </select>

      <label>Other Debtors / Guarantors:</label>
      <select
        name="OtherDebtors"
        value={formData.OtherDebtors}
        onChange={handleChange}
      >
        <option value="A101">None</option>
        <option value="A102">Co-applicant</option>
        <option value="A103">Guarantor</option>
      </select>

      <label>Property:</label>
      <select
        name="Property"
        value={formData.Property}
        onChange={handleChange}
      >
        <option value="A121">Real estate</option>
        <option value="A122">Savings insurance</option>
        <option value="A123">Car or other</option>
        <option value="A124">Unknown / No property</option>
      </select>

      <label>Other Installment Plans:</label>
      <select
        name="OtherInstallmentPlans"
        value={formData.OtherInstallmentPlans}
        onChange={handleChange}
      >
        <option value="A141">Bank</option>
        <option value="A142">Stores</option>
        <option value="A143">None</option>
      </select>

      <label>Housing:</label>
      <select name="Housing" value={formData.Housing} onChange={handleChange}>
        <option value="A151">Rent</option>
        <option value="A152">Own</option>
        <option value="A153">For free</option>
      </select>

      <label>Job:</label>
      <select name="Job" value={formData.Job} onChange={handleChange}>
        <option value="A171">Unemployed / unskilled (non-resident)</option>
        <option value="A172">Unskilled (resident)</option>
        <option value="A173">Skilled employee / official</option>
        <option value="A174">Management / self-employed</option>
      </select>

      <label>Telephone:</label>
      <select
        name="Telephone"
        value={formData.Telephone}
        onChange={handleChange}
      >
        <option value="A191">None</option>
        <option value="A192">Yes, registered under customer’s name</option>
      </select>

      <label>Foreign Worker:</label>
      <select
        name="ForeignWorker"
        value={formData.ForeignWorker}
        onChange={handleChange}
      >
        <option value="A201">Yes</option>
        <option value="A202">No</option>
      </select>

      <button type="submit" disabled={loading}>
          {loading ? "Predicting..." : "Predict Credit Risk"}
        </button>
      </form>

      {error && <div className="error">⚠️ Error: {error}</div>}

      {result && (
  <div
    className={`result-card ${
      result.label.includes("Good") ? "good" : "bad"
    }`}
  >
    <h3>Prediction Result</h3>
    <p>
      <strong>Label:</strong> {result.label}
    </p>
    <p>
      <strong>Prediction:</strong>{" "}
      {result.label.includes("Good") ? "✅ Good Credit" : "❌ Bad Credit"}
    </p>
    <p>
      <strong>Risk Probability (bad):</strong>{" "}
      {result.risk_probability.toFixed(2)}
    </p>
    {/* ✅ Show Explain button only if we have a result */}
    <button onClick={handleExplain} disabled={loading} className="mt-3">
            {loading ? "Explaining..." : "Explain Prediction"}
          </button>
  </div>
)}
{/* ✅ Chart below result */}
{explanation && <ExplanationChart explanation={explanation} />}
    </div>
  );
}

export default CreditForm;