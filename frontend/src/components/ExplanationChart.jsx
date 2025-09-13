import React from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Cell,
} from "recharts";
import { Card, CardHeader, CardTitle, CardContent } from "./ui/card";

// Complete mapping for all features (German Credit dataset)
const featureNameMap = {
  Duration: "Loan Duration (months)",
  CreditAmount: "Credit Amount",
  InstallmentRate: "Installment Rate (% of income)",
  Age: "Age (years)",
  ResidenceDuration: "Residence Duration (years)",
  ExistingCredits: "Number of Existing Credits",
  LiableDependents: "Number of Dependents",

  // Status (checking account)
  "Status = A11": "No checking account",
  "Status = A12": "0 ≤ ... < 200 DM",
  "Status = A13": "≥ 200 DM",
  "Status = A14": "Checking account not available",

  // Credit history
  "CreditHistory = A30": "No credit taken / all credits paid",
  "CreditHistory = A31": "All credits paid duly",
  "CreditHistory = A32": "Existing credits paid properly",
  "CreditHistory = A33": "Delay in past payments",
  "CreditHistory = A34": "Critical account / other credits existing",

  // Purpose
  "Purpose = A40": "Car (new)",
  "Purpose = A41": "Car (used)",
  "Purpose = A42": "Furniture / Equipment",
  "Purpose = A43": "Radio / TV",
  "Purpose = A44": "Domestic appliances",
  "Purpose = A45": "Repairs",
  "Purpose = A46": "Education",
  "Purpose = A47": "Vacation",
  "Purpose = A48": "Retraining",
  "Purpose = A49": "Business",
  "Purpose = A410": "Other",

  // Savings
  "Savings = A61": "< 100 DM",
  "Savings = A62": "100 ≤ ... < 500 DM",
  "Savings = A63": "500 ≤ ... < 1000 DM",
  "Savings = A64": "≥ 1000 DM",
  "Savings = A65": "Unknown / None",

  // Employment
  "Employment = A71": "Unemployed",
  "Employment = A72": "< 1 year",
  "Employment = A73": "1 ≤ ... < 4 years",
  "Employment = A74": "4 ≤ ... < 7 years",
  "Employment = A75": "≥ 7 years",

  // Personal status & sex
  "PersonalStatusSex = A91": "Male: divorced/separated",
  "PersonalStatusSex = A92": "Female: divorced/separated/married",
  "PersonalStatusSex = A93": "Male: single",
  "PersonalStatusSex = A94": "Male: married/widowed",
  "PersonalStatusSex = A95": "Female: single",

  // Other debtors / guarantors
  "OtherDebtors = A101": "None",
  "OtherDebtors = A102": "Co-applicant",
  "OtherDebtors = A103": "Guarantor",

  // Property
  "Property = A121": "Real estate",
  "Property = A122": "Savings / Insurance",
  "Property = A123": "Car or other assets",
  "Property = A124": "Unknown / None",

  // Other installment plans
  "OtherInstallmentPlans = A141": "Bank",
  "OtherInstallmentPlans = A142": "Stores",
  "OtherInstallmentPlans = A143": "None",

  // Housing
  "Housing = A151": "Rent",
  "Housing = A152": "Own",
  "Housing = A153": "Free",

  // Job
  "Job = A171": "Unemployed / unskilled (non-resident)",
  "Job = A172": "Unskilled (resident)",
  "Job = A173": "Skilled employee / official",
  "Job = A174": "Management / self-employed",

  // Telephone
  "Telephone = A191": "None",
  "Telephone = A192": "Yes (registered)",

  // Foreign worker
  "ForeignWorker = A201": "Yes",
  "ForeignWorker = A202": "No",
};

export default function ExplanationChart({ explanation }) {
  if (!explanation) return null;

  const { label, risk_probability, top_features } = explanation;

  // Convert SHAP features to recharts-friendly format
  const data = Object.entries(top_features).map(([feature, value]) => ({
    feature: featureNameMap[feature] || feature, // map or fallback
    impact: value,
  }));

  return (
    <Card className="w-full max-w-2xl shadow-md rounded-2xl border p-4">
      <CardHeader>
        <CardTitle className="text-xl font-semibold">
          Prediction:{" "}
          <span
            className={
              label === "Good Credit" ? "text-green-600" : "text-red-600"
            }
          >
            {label}
          </span>
        </CardTitle>
        <p className="text-sm text-gray-500">
          Risk Probability: {(risk_probability * 100).toFixed(1)}%
        </p>
      </CardHeader>

      <CardContent>
        <h3 className="text-base font-medium mb-3">
          Top Contributing Features
        </h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart
            layout="vertical"
            data={data.sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact))}
            margin={{ top: 10, right: 20, left: 120, bottom: 10 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis dataKey="feature" type="category" width={220} />
            <Tooltip
              formatter={(value) => value.toFixed(3)}
              labelStyle={{ fontWeight: "bold" }}
            />
            <Bar dataKey="impact" barSize={18} radius={[4, 4, 4, 4]}>
              {data.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={entry.impact >= 0 ? "#22c55e" : "#ef4444"} // green = supports good, red = supports bad
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>

        {/* Legend */}
        <div className="flex items-center justify-center gap-6 mt-4 text-sm">
          <div className="flex items-center gap-2">
            <span
              className="inline-block w-4 h-4 rounded"
              style={{ backgroundColor: "#22c55e" }}
            ></span>
            <span>Supports Good Credit</span>
          </div>
          <div className="flex items-center gap-2">
            <span
              className="inline-block w-4 h-4 rounded"
              style={{ backgroundColor: "#ef4444" }}
            ></span>
            <span>Supports Bad Credit</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}