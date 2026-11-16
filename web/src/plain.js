// Plain-language formatting shared by every screen (same rules as procintel/monitoring.py plain_duration / grade_gap).

// 43 -> "43 days", 547 -> "18 months", 900 -> "2.5 years"
export function dur(days) {
  if (days == null || Number.isNaN(Number(days))) return "–";
  const d = Math.max(0, Math.round(Number(days)));
  if (d < 60) return `${d} day${d === 1 ? "" : "s"}`;
  const m = d / 30.44;
  if (m < 23.5) return `${Math.round(m)} months`;
  return `${Math.round((m / 12) * 2) / 2} years`;
}

// A change in days ("+12" -> "adds about 2 weeks", "-40" -> "saves about 1 month")
export function durDelta(days) {
  const d = Math.round(Number(days) || 0), a = Math.abs(d);
  const size = a < 4 ? "a few days" : a < 25 ? `about ${Math.max(1, Math.round(a / 7))} week${Math.round(a / 7) > 1 ? "s" : ""}` : `about ${dur(a)}`;
  return a < 1 ? "no change" : `${d > 0 ? "adds" : "saves"} ${size}`;
}

// ISR-vs-ICR rating gap (1-6 scale) in words
export function gradeGap(gap) {
  const a = Math.abs(gap);
  const size = a < 0.35 ? "slightly" : a < 0.75 ? "about half a grade" : a < 1.25 ? "about one grade" : `about ${Math.round(a)} grades`;
  return `${size} ${gap > 0 ? "higher" : "lower"}`;
}

// What each risk level means (scores: rating warnings 0.3 each, late tender / slow spending 0.25 each, capped at 1)
export const RISK_HELP = {
  "High risk": "Several warning signs together, e.g. a dropped and a low progress rating, or a rating warning plus tenders running late.",
  Watch: "A progress-rating warning, or two smaller warning signs.",
  Minor: "One smaller warning sign, such as one tender running late.",
  "No warning signs": "No warning signs in the data.",
};

// Model inputs in plain words (procintel/models.py CAT + NUM)
export const FEATURE = {
  category: "procurement category", procurement_method: "procurement method", review_type: "type of Bank review", sector: "sector",
  country_code: "country", es_risk: "environmental & social risk", log_value: "contract value", wgi_rule_of_law: "rule of law in the country",
  wgi_gov_effectiveness: "government effectiveness", wgi_reg_quality: "regulatory quality", cpia: "country policy rating (CPIA)",
  notice_month: "month the tender is advertised",
};
