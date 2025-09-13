import React from "react";

export function Card({ children, className }) {
  return (
    <div className={`bg-white shadow rounded-xl p-4 ${className || ""}`}>
      {children}
    </div>
  );
}

export function CardHeader({ children }) {
  return <div className="mb-3 border-b pb-2">{children}</div>;
}

export function CardTitle({ children, className }) {
  return (
    <h2 className={`text-lg font-semibold ${className || ""}`}>{children}</h2>
  );
}

export function CardContent({ children }) {
  return <div>{children}</div>;
}