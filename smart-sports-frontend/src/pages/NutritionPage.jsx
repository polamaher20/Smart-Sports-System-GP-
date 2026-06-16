import { useState, useEffect } from "react";
import { useNavigate } from 'react-router-dom';
import { generatePlan } from '../services/api';

const ACTIVITY_LEVELS = [
  { value: "sedentary", label: "Sedentary", desc: "Little or no exercise" },
  { value: "light",     label: "Light",     desc: "Exercise 1-3 days/week" },
  { value: "moderate",  label: "Moderate",  desc: "Exercise 3-5 days/week" },
  { value: "active",    label: "Active",    desc: "Exercise 6-7 days/week" },
  { value: "athlete",   label: "Athlete",   desc: "Very intense training or professional athlete" },
];

const GOALS = [
  { value: "lose_weight",  label: "Lose Weight",  icon: "🔥", color: "#ef4444" },
  { value: "maintain",     label: "Maintain",     icon: "⚖️", color: "#f97316" },
  { value: "gain_muscle",  label: "Gain Muscle",  icon: "💪", color: "#2563eb" },
];

const JOB_TYPES = [
  "Office Worker",
  "Student",
  "Manual Labor",
  "Engineer",
  "Medical Staff",
  "Trainer",
  "Freelancer",
];

export default function NutritionPage() {
  const [step, setStep] = useState(1);
  const [form, setForm] = useState({
    age: "",
    weight: "",
    height: "",
    gender: "male",
    activity: "moderate",
    goal: "maintain",
    job: "Office Worker",

    // الحقول الجديدة
    water_intake_liters: "2",
    meals_per_day: "3",
    has_snacks: "1",
    surgery_type: [],

    // Allergies
    nut_allergy: "0",
    dairy_allergy: "0",
    gluten_intolerance: "0",
    shellfish_allergy: "0",
    egg_allergy: "0",

    // Sport related (هتظهر conditionally)
    sport_type: "None",
    experience_level: "Beginner",
    training_days_per_week: "0",
    training_duration_min: "0",

    // Injuries
    ankle_injury: "0",
    back_pain: "0",
    muscle_tear: "0",

    // Diseases
    diabetes: "0",
    hypertension: "0",
    kidney_disease: "0",
    heart_disease: "0",
    pcos: "0",
    anemia: "0",
    gout: "0",
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState({});
  const [showMedical, setShowMedical] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    const userData = JSON.parse(localStorage.getItem('user') || '{}');
    if (userData) {
      setForm(prev => ({
        ...prev,
        age:    userData.age      ? String(userData.age)      : "",
        weight: userData.weight_kg ? String(userData.weight_kg) : "",
        height: userData.height_cm ? String(userData.height_cm) : "",
        gender: userData.gender === "Female" ? "female" : "male",
        sport_type: userData.sport || "None",
        surgery_type: Array.isArray(userData.surgery_type)
          ? userData.surgery_type.filter(v => v && v !== "None")
          : userData.surgery_type
            ? String(userData.surgery_type).split(",").map(s => s.trim()).filter(v => v && v !== "None")
            : [],
        nut_allergy: userData.nut_allergy ? String(userData.nut_allergy) : "0",
        dairy_allergy: userData.dairy_allergy ? String(userData.dairy_allergy) : "0",
        gluten_intolerance: userData.gluten_intolerance ? String(userData.gluten_intolerance) : "0",
        shellfish_allergy: userData.shellfish_allergy ? String(userData.shellfish_allergy) : "0",
        egg_allergy: userData.egg_allergy ? String(userData.egg_allergy) : "0",
        ankle_injury: userData.ankle_injury ? String(userData.ankle_injury) : "0",
        back_pain: userData.back_pain ? String(userData.back_pain) : "0",
        muscle_tear: userData.muscle_tear ? String(userData.muscle_tear) : "0",
        diabetes: userData.diabetes ? String(userData.diabetes) : "0",
        hypertension: userData.hypertension ? String(userData.hypertension) : "0",
        kidney_disease: userData.kidney_disease ? String(userData.kidney_disease) : "0",
        heart_disease: userData.heart_disease ? String(userData.heart_disease) : "0",
        pcos: userData.pcos ? String(userData.pcos) : "0",
        anemia: userData.anemia ? String(userData.anemia) : "0",
        gout: userData.gout ? String(userData.gout) : "0",
        training_days_per_week: userData.training_days_per_week ? String(userData.training_days_per_week) : "0",
        training_duration_min: userData.training_duration_min ? String(userData.training_duration_min) : "0",
        job: userData.job ? String(userData.job) : "Office Worker",
      }));
    }
  }, []);

  const set = (k, v) => {
    setForm(p => ({ ...p, [k]: v }));
    setErrors(p => ({ ...p, [k]: "" }));
  };

  const validate = () => {
    const e = {};
    if (!form.age    || form.age < 10 || form.age > 100) e.age    = "Enter valid age (10-100)";
    if (!form.weight || form.weight < 30)                 e.weight = "Enter valid weight";
    if (!form.height || form.height < 100)                e.height = "Enter valid height";
    return e;
  };

  const calculate = async () => {
    const e = validate();
    if (Object.keys(e).length > 0) { setErrors(e); return; }

    setLoading(true);
    try {
      const weight = +form.weight;
      const heightM = +form.height / 100;
      const bmi = weight / (heightM * heightM);

      const payload = {
        age: Number(form.age),
        gender: form.gender === "male" ? "Male" : "Female",
        weight_kg: Number(form.weight),
        height_cm: Number(form.height),
        BMI: Math.round((Number(form.weight) / Math.pow(Number(form.height)/100, 2)) * 10) / 10,

        goal: form.goal === "lose_weight" ? "Fat_Loss"
          : form.goal === "gain_muscle" ? "Muscle_Gain"
          : "Maintenance",

        activity: form.activity === "sedentary" ? "Sedentary"
          : form.activity === "light" ? "Light"
          : form.activity === "moderate" ? "Moderate"
          : form.activity === "active" ? "Active"
          : "Moderate",

        job: form.job,
        surgery_type: form.surgery_type || [],
        sport_type: form.sport_type || "None",

        diabetes: Number(form.diabetes || 0),
        hypertension: Number(form.hypertension || 0),
        kidney_disease: Number(form.kidney_disease || 0),
        heart_disease: Number(form.heart_disease || 0),
        pcos: Number(form.pcos || 0),
        anemia: Number(form.anemia || 0),
        gout: Number(form.gout || 0),

        ankle_injury: Number(form.ankle_injury || 0),
        back_pain: Number(form.back_pain || 0),
        muscle_tear: Number(form.muscle_tear || 0),
      };

      console.log("📤 Sending Payload:", payload);
      const res = await generatePlan(payload);
      console.log("✅ Response from Backend:", res);
      setResult(res.data || res);
      setStep(3);
    } catch (err) {
      console.error(err);
      const errorData = err.response?.data;
      if (errorData?.detail) {
        let message = "Validation Errors:\n";
        if (Array.isArray(errorData.detail)) {
          errorData.detail.forEach((er, i) => { message += `${i+1}. ${JSON.stringify(er)}\n`; });
        } else message += JSON.stringify(errorData.detail);
        alert(message);
        setErrors({ general: message });
      } else {
        alert("حدث خطأ غير معروف");
      }
    } finally {
      setLoading(false);
    }
  };

  const goalInfo = GOALS.find(g => g.value === form.goal);

  return (
    <div style={{ minHeight:"100vh", background:"#f5f7fa", fontFamily:"Segoe UI,sans-serif" }}>
      <style>{`
        * { box-sizing:border-box; margin:0; padding:0; }
        .fi { width:100%; padding:.7rem .9rem; border:1.5px solid #e2eaf2; border-radius:10px; font-size:.88rem; font-family:inherit; background:white; color:#1a2332; outline:none; transition:border-color .2s; }
        .fi:focus { border-color:#2563eb; box-shadow:0 0 0 3px rgba(37,99,235,.08); }
        .fi.err { border-color:#ef4444; }
        .fi::placeholder { color:#cbd5e1; }
        .goal-card { border:2px solid #e2eaf2; border-radius:14px; padding:1.1rem; cursor:pointer; transition:all .2s; background:white; text-align:center; flex:1; }
        .goal-card:hover { border-color:#2563eb; transform:translateY(-2px); }
        .goal-card.on { border-color:#2563eb; background:#eff6ff; }
        .act-card { border:1.5px solid #e2eaf2; border-radius:10px; padding:.85rem 1rem; cursor:pointer; transition:all .2s; background:white; display:flex; align-items:center; gap:.75rem; }
        .act-card:hover { border-color:#2563eb; }
        .act-card.on { border-color:#2563eb; background:#eff6ff; }
        .btn-orange { background:#f97316; color:white; border:none; padding:.8rem 2rem; border-radius:10px; font-weight:700; font-size:.95rem; cursor:pointer; transition:all .2s; }
        .btn-orange:hover { background:#ea580c; transform:translateY(-1px); }
        .btn-outline { background:white; color:#64748b; border:1.5px solid #e2eaf2; padding:.75rem 2rem; border-radius:10px; font-weight:600; font-size:.9rem; cursor:pointer; transition:all .2s; }
        .btn-outline:hover { border-color:#2563eb; color:#2563eb; }
        .macro-card { background:white; border:1px solid #e2eaf2; border-radius:16px; padding:1.5rem; text-align:center; flex:1; transition:all .2s; }
        .macro-card:hover { box-shadow:0 8px 24px rgba(37,99,235,.1); transform:translateY(-3px); }
        .step-dot { width:32px; height:32px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-weight:700; font-size:.8rem; transition:all .3s; }
        @keyframes fadeUp { from{opacity:0;transform:translateY(20px)} to{opacity:1;transform:translateY(0)} }
        .fu { animation:fadeUp .5s ease both; }
      `}</style>

      {/* Navbar */}
      <nav style={{ background:"white", borderBottom:"1px solid #e2eaf2", padding:".85rem 2rem", display:"flex", flexWrap:"wrap" , justifyContent:"center", alignItems:"center", gap:"1rem" }}>
        <div style={{ display:"flex", alignItems:"center", gap:".5rem" }}>
          <div style={{ width:32, height:32, background:"linear-gradient(135deg,#1e3a5f,#f97316)", borderRadius:"50%", display:"flex", alignItems:"center", justifyContent:"center" }}>
            <div style={{ width:16, height:16, borderRadius:"50%", border:"2px solid white" }} />
          </div>
          <span style={{ fontWeight:800, fontSize: ".9rem", color:"#1a2332" }}>Sportiva</span>
        </div>
        <div style={{ marginLeft:"auto", display:"flex", gap:"2rem" }}>
          {[{ label: "Home", path: "/" }, { label: "Fitness", path: "/nutrition" }, { label: "Performance", path: "/dashboard" }, { label: "Interview", path: "/dashboard" }].map(l => (
            <span key={l.label} onClick={() => navigate(l.path)} style={{ fontSize: ".85rem", color: "#64748b", cursor: "pointer", fontWeight:500 }}>{l.label}</span>
          ))}
        </div>
      </nav>

      <div style={{ maxWidth:720, margin:"0 auto", padding:"2.5rem 1.5rem" }}>

        {/* Header */}
        <div style={{ textAlign:"center", marginBottom:"2rem" }}>
          <div style={{ width:56, height:56, background:"linear-gradient(135deg,#2563eb,#1d4ed8)", borderRadius:"50%", display:"flex", alignItems:"center", justifyContent:"center", fontSize:"1.4rem", margin:"0 auto .9rem" }}>🥗</div>
          <h1 style={{ fontWeight:800, fontSize:"1.6rem", color:"#1a2332", marginBottom: ".4rem" }}>Nutrition Plan Generator</h1>
          <p style={{ color:"#64748b", fontSize:".88rem" }}>Get your personalized daily nutrition targets based on your body and goals</p>
        </div>

        {/* Steps indicator */}
        <div style={{ display:"flex", alignItems:"center", justifyContent:"center", gap:0, marginBottom:"2rem" }}>
          { ["Your Info","Your Goal","Results"].map((s,i) => (
            <div key={s} style={{ display:"flex", alignItems:"center" }}>
              <div style={{ display:"flex", flexDirection:"column", alignItems:"center", gap: ".3rem" }}>
                <div className="step-dot" style={{ background: step > i+1 ? "#10b981" : step === i+1 ? "#2563eb" : "#e2eaf2", color: step >= i+1 ? "white" : "#94a3b8" }}>{step > i+1 ? "✓" : i+1}</div>
                <span style={{ fontSize: ".68rem", color: step === i+1 ? "#2563eb" : "#94a3b8", fontWeight: step === i+1 ? 700 : 400 }}>{s}</span>
              </div>
              {i < 2 && <div style={{ width:60, height:2, background: step > i+1 ? "#10b981" : "#e2eaf2", margin:"0 .5rem", marginBottom:"1.1rem" }} />}
            </div>
          ))}
        </div>

        {/* ── STEP 1: Personal Info ── */}
        {step === 1 && (
          <div className="fu" style={{ background:"white", border:"1px solid #e2eaf2", borderRadius:20, padding:"2rem" }}>
            <h2 style={{ fontWeight:700, fontSize:"1.1rem", color:"#1a2332", marginBottom:"1.5rem" }}>Tell us about yourself</h2>

            {/* Basic Info */}
            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:"1rem", marginBottom:"1.5rem" }}>
              <div>
                <label>Age (years)</label>
                <input className={`fi${errors.age?" err":""}`} type="number" value={form.age} onChange={e => set("age", e.target.value)} />
              </div>
              <div>
                <label>Gender</label>
                <select className="fi" value={form.gender} onChange={e => set("gender", e.target.value)}>
                  <option value="male">Male</option>
                  <option value="female">Female</option>
                </select>
              </div>
              <div>
                <label>Weight (kg)</label>
                <input className={`fi${errors.weight?" err":""}`} type="number" value={form.weight} onChange={e => set("weight", e.target.value)} />
              </div>
              <div>
                <label>Height (cm)</label>
                <input className={`fi${errors.height?" err":""}`} type="number" value={form.height} onChange={e => set("height", e.target.value)} />
              </div>
            </div>

            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:"1rem", marginBottom:"1.5rem" }}>
              <div>
                <label>Activity Level</label>
                <select className="fi" value={form.activity} onChange={e => set("activity", e.target.value)}>
                  {ACTIVITY_LEVELS.map(item => (<option key={item.value} value={item.value}>{item.label}</option>))}
                </select>
              </div>
              <div>
                <label>Activity Description</label>
                <div className="fi" style={{ minHeight: "3.2rem", display:"flex", alignItems:"center", color:"#475569" }}>
                  {ACTIVITY_LEVELS.find(item => item.value === form.activity)?.desc || "Select your activity level"}
                </div>
              </div>
            </div>

            {/* New Fields */}
            <div style={{ marginTop: "2rem" }}>
              <h3 style={{ marginBottom: "1rem", fontSize: "1rem" }}>Additional Information</h3>

              <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:"1rem" }}>
                <div>
                  <label>Water Intake (Liters/day)</label>
                  <input className="fi" type="number" step="0.5" value={form.water_intake_liters} onChange={e => set("water_intake_liters", e.target.value)} />
                </div>
                <div>
                  <label>Meals per Day</label>
                  <select className="fi" value={form.meals_per_day} onChange={e => set("meals_per_day", e.target.value)}>
                    {[3,4,5,6].map(n => <option key={n} value={n}>{n} meals</option>)}
                  </select>
                </div>
              </div>

              <div style={{ marginTop: "1rem" }}>
                <label>Has Snacks?</label>
                <select className="fi" value={form.has_snacks} onChange={e => set("has_snacks", e.target.value)}>
                  <option value="1">Yes</option>
                  <option value="0">No</option>
                </select>
              </div>

              {/* Medical Info Accordion */}
              <div style={{ marginTop: "1rem" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <div style={{ fontWeight: 800, color: "#1a2332" }}>Medical Information</div>
                  <button className="btn-outline" onClick={() => setShowMedical(s => !s)} style={{ padding: ".4rem .75rem" }}>{showMedical ? "Hide" : "Show"}</button>
                </div>

                {showMedical && (
                  <div style={{ marginTop: "1rem" }}>
                    <div style={{ marginTop: "1rem" }}>
                      <label style={{ display:"block", marginBottom:'.5rem', fontWeight:600 }}>Injuries</label>
                      <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fit, minmax(140px, 1fr))", gap:".75rem" }}>
                        {[{key: "ankle_injury", label: "Ankle"}, {key: "back_pain", label: "Back pain"}, {key: "muscle_tear", label: "Muscle tear"}].map(item => (
                          <label key={item.key} style={{ display:"flex", alignItems:"center", gap:".5rem", fontSize:".85rem" }}>
                            <input type="checkbox" checked={form[item.key] === "1"} onChange={e => set(item.key, e.target.checked ? "1" : "0")} />
                            {item.label}
                          </label>
                        ))}
                      </div>
                    </div>

                    <div style={{ marginTop: "1rem" }}>
                      <label style={{ display:"block", marginBottom:".5rem", fontWeight:600 }}>Medical Conditions</label>
                      <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fit, minmax(140px, 1fr))", gap:".75rem" }}>
                        {[{key: "diabetes", label: "Diabetes"}, {key: "hypertension", label: "Hypertension"}, {key: "kidney_disease", label: "Kidney disease"}, {key: "heart_disease", label: "Heart disease"}, {key: "pcos", label: "PCOS"}, {key: "anemia", label: "Anemia"}, {key: "gout", label: "Gout"}].map(item => (
                          <label key={item.key} style={{ display:"flex", alignItems:"center", gap:".5rem", fontSize:".85rem" }}>
                            <input type="checkbox" checked={form[item.key] === "1"} onChange={e => set(item.key, e.target.checked ? "1" : "0")} />
                            {item.label}
                          </label>
                        ))}
                      </div>
                    </div>

                    <div style={{ marginTop: "1rem" }}>
                      <label style={{ display:"block", marginBottom:".5rem", fontWeight:600 }}>Surgery Type</label>
                      <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fit, minmax(160px, 1fr))", gap:"0.75rem" }}>
                        {[{value: "Gastric Bypass", label: "Gastric Bypass"}, {value: "Sleeve Gastrectomy", label: "Sleeve Gastrectomy"}, {value: "Gallbladder Removal", label: "Gallbladder Removal"}, {value: "ACL Surgery", label: "ACL Surgery"}, {value: "Spinal Surgery", label: "Spinal Surgery"}, {value: "Thyroid Surgery", label: "Thyroid Surgery"}].map(item => (
                          <label key={item.value} style={{ display:"flex", alignItems:"center", gap:"0.5rem", fontSize:"0.95rem" }}>
                            <input type="checkbox" checked={form.surgery_type.includes(item.value)} onChange={e => {
                              const next = e.target.checked ? [...form.surgery_type, item.value] : form.surgery_type.filter(v => v !== item.value);
                              set("surgery_type", next);
                            }} />
                            {item.label}
                          </label>
                        ))}
                      </div>
                    </div>

                    <div style={{ marginTop: "1.5rem" }}>
                      <label style={{ display:"block", marginBottom:".5rem", fontWeight:600 }}>Allergies / Intolerances</label>
                      <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fit, minmax(140px, 1fr))", gap:".75rem" }}>
                        {[{key: "nut_allergy", label: "Nuts"}, {key: "dairy_allergy", label: "Dairy"}, {key: "gluten_intolerance", label: "Gluten"}, {key: "shellfish_allergy", label: "Shellfish"}, {key: "egg_allergy", label: "Eggs"}].map(item => (
                          <label key={item.key} style={{ display:"flex", alignItems:"center", gap:".5rem", fontSize:".85rem" }}>
                            <input type="checkbox" checked={form[item.key] === "1"} onChange={e => set(item.key, e.target.checked ? "1" : "0")} />
                            {item.label}
                          </label>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Job (moved before sport) */}
              <div style={{ marginTop: "1rem" }}>
                <label>Job Type</label>
                <select className="fi" value={form.job} onChange={e => set("job", e.target.value)}>
                  {JOB_TYPES.map(job => <option key={job} value={job}>{job}</option>)}
                </select>
              </div>

              {/* Sport Section */}
              <div style={{ marginTop: "1.5rem" }}>
                <label>Sport Type</label>
                <select className="fi" value={form.sport_type} onChange={e => set("sport_type", e.target.value)}>
                  <option value="None">None</option>
                  <option value="Football">Football</option>
                  <option value="Bodybuilding">Bodybuilding</option>
                  <option value="Running">Running</option>
                  <option value="Yoga">Yoga</option>
                </select>

                {form.sport_type !== "None" && (
                  <>
                    <div style={{ marginTop: "1rem" }}>
                      <label>Experience Level</label>
                      <select className="fi" value={form.experience_level} onChange={e => set("experience_level", e.target.value)}>
                        <option value="Beginner">Beginner</option>
                        <option value="Intermediate">Intermediate</option>
                        <option value="Professional">Professional</option>
                      </select>
                    </div>

                    <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:"1rem", marginTop:"1rem" }}>
                      <div>
                        <label>Training Days / Week</label>
                        <input className="fi" type="number" min="1" max="7" value={form.training_days_per_week} onChange={e => set("training_days_per_week", e.target.value)} />
                      </div>
                      <div>
                        <label>Duration (minutes)</label>
                        <input className="fi" type="number" value={form.training_duration_min} onChange={e => set("training_duration_min", e.target.value)} />
                      </div>
                    </div>
                  </>
                )}
              </div>
            </div>

            <button className="btn-orange" style={{ width:"100%", marginTop:"2rem" }} onClick={() => {
              const e = validate(); if (Object.keys(e).length > 0) { setErrors(e); return; } setStep(2);
            }}>
              Next: Choose Your Goal →
            </button>
          </div>
        )}

        {/* STEP 2 + STEP 3 sections unchanged (kept original rendering) */}
        {step === 2 && (
          <div className="fu" style={{ background:"white", border:"1px solid #e2eaf2", borderRadius:20, padding:"2rem" }}>
            <h2 style={{ fontWeight:700, fontSize:"1.1rem", color:"#1a2332", marginBottom:".4rem" }}>What is your goal?</h2>
            <p style={{ color:"#64748b", fontSize:" .82rem", marginBottom:"1.5rem" }}>Choose the goal that best describes what you want to achieve</p>

            <div style={{ display:"flex", gap:"1rem", marginBottom:"2rem" }}>
              {GOALS.map(g => (
                <div key={g.value} className={"goal-card" + (form.goal===g.value?" on":"")} onClick={() => set("goal", g.value)}>
                  <div style={{ fontSize:"2rem", marginBottom: ".5rem" }}>{g.icon}</div>
                  <div style={{ fontWeight:700, fontSize:".88rem", color:"#1a2332" }}>{g.label}</div>
                  <div style={{ width:16, height:16, borderRadius:"50%", border:`2px solid ${form.goal===g.value?g.color:"#e2eaf2"}`, margin: ".75rem auto 0", display:"flex", alignItems:"center", justifyContent:"center" }}>
                    {form.goal===g.value && <div style={{ width:8, height:8, borderRadius:"50%", background:g.color }} />}
                  </div>
                </div>
              ))}
            </div>

            <div style={{ background:"#f8fafc", border:"1px solid #e2eaf2", borderRadius:12, padding:"1rem", marginBottom:"1.5rem" }}>
              <div style={{ fontSize: ".75rem", fontWeight:600, color:"#64748b", marginBottom: ".6rem" }}>Your Summary</div>
              <div style={{ display:"flex", gap:"1.5rem", flexWrap:"wrap" }}>
                {[ ["Age", form.age+" yrs"],["Weight",form.weight+" kg"],["Height",form.height+" cm"],["Activity",form.activity],["Goal",goalInfo?.label] ].map(([l,v]) => (
                  <div key={l}><div style={{ fontSize: ".65rem", color: "#94a3b8" }}>{l}</div><div style={{ fontWeight:700, fontSize: ".82rem", color: "#1a2332", textTransform: "capitalize" }}>{v}</div></div>
                ))}
              </div>
            </div>

            <div style={{ display:"flex", gap:"1rem" }}>
              <button className="btn-outline" onClick={() => setStep(1)}>← Back</button>
              <button className="btn-orange" style={{ flex:1 }} onClick={calculate} disabled={loading}>{loading ? "Calculating..." : "Generate My Plan 🚀"}</button>
            </div>
          </div>
        )}

        {step === 3 && result && (
          <div className="fu" style={{ background:"white", border:"1px solid #e2eaf2", borderRadius:20, padding:"2rem" }}>
            {/* Results rendering (kept original) */}
            <div style={{ background:"linear-gradient(135deg,#1e3a5f,#2563eb)", borderRadius:20, padding:"2rem", textAlign:"center", marginBottom:"1.5rem", color:"white" }}>
              <div style={{ fontSize:"3.8rem", fontWeight:900 }}>{result.calories || result.daily_calories}</div>
              <div style={{ fontSize: ".9rem", opacity:0.9 }}>Calories / Day</div>
            </div>

            <div style={{ display:"flex", gap:"1rem", marginBottom:"2rem" }}>
              {[
                { label: "Protein", grams: result.protein_grams, percent: result.protein_percent || result.daily_macros?.protein },
                { label: "Carbs", grams: result.carbs_grams, percent: result.carbs_percent || result.daily_macros?.carbs },
                { label: "Fat", grams: result.fat_grams, percent: result.fat_percent || result.daily_macros?.fat },
              ].map(m => (
                <div key={m.label} style={{ flex:1, background:"#f8fafc", borderRadius:16, padding:"1.25rem", textAlign:"center" }}>
                  <div style={{ fontSize:"2.1rem", fontWeight:800, color:"#1e40af" }}>{Math.round(m.grams)}g</div>
                  <div style={{ fontWeight:600, margin:"0.4rem 0" }}>{m.label}</div>
                  <div style={{ fontSize:"0.95rem", color:"#64748b" }}>{Math.round(m.percent)}%</div>
                </div>
              ))}
            </div>

            {/* Weekly Meal Plan & Workout display kept similar to original */}
            {result.weekly_plan && result.weekly_plan.days && (
              <div style={{ marginBottom: "2rem" }}>
                <h3 style={{ marginBottom: "1rem" }}>📅 Weekly Meal Plan</h3>
                {result.weekly_plan.days.map((day, i) => (
                  <div key={i} style={{ marginBottom: "1.5rem", padding: "1rem", background: "#f8fafc", borderRadius: 12 }}>
                    <h4 style={{ color: "#1e40af", marginBottom: "0.8rem" }}>{day.day}</h4>
                    {day.meals?.map((meal, j) => (
                      <div key={j} style={{ marginBottom: "1rem" }}>
                        <strong>🍽️ {meal.name}</strong>
                        <ul style={{ fontSize: "0.9rem", color: "#475569", marginTop: "0.4rem" }}>
                          {meal.foods?.map((food, k) => (<li key={k}>• {food.name} — {food.amount_g}g</li>))}
                        </ul>
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            )}

            {result.workout && result.workout.weekly_schedule && (
              <div style={{ background:"white", border:"1px solid #e2eaf2", borderRadius:16, padding:"1.5rem", marginBottom:"1.25rem" }}>
                <h3 style={{ fontWeight:700, fontSize: ".9rem", color: "#1a2332", marginBottom:"1rem" }}>💪 Weekly Workout Plan</h3>
                {result.workout.weekly_schedule.map((day, di) => (
                  <div key={di} style={{ marginBottom:"1rem", borderBottom:"1px solid #f0f4f8", paddingBottom:"1rem" }}>
                    <div style={{ fontWeight:700, fontSize: ".85rem", color: "#f97316", marginBottom: ".5rem" }}>📅 {day.day}</div>
                    {day.exercises.map((ex, ei) => (
                      <div key={ei} style={{ padding: ".4rem 0 .4rem 1rem", borderBottom: "1px solid #f8fafc" }}>
                        <div style={{ fontWeight:600, fontSize: ".78rem", color: "#1a2332" }}>• {ex.name}</div>
                        <div style={{ fontSize: ".7rem", color: "#64748b" }}>{ex.sets} sets × {ex.reps} | Rest: {ex.rest}</div>
                      </div>
                    ))}
                  </div>
                ))}
                {result.workout.tips && (
                  <div style={{ marginTop: "1rem" }}>
                    <div style={{ fontWeight:600, fontSize: ".78rem", color: "#1a2332", marginBottom: ".5rem" }}>💡 Tips:</div>
                    {result.workout.tips.map((tip, i) => (<div key={i} style={{ fontSize: ".75rem", color: "#64748b", padding: ".2rem 0" }}>• {tip}</div>))}
                  </div>
                )}
              </div>
            )}

            <div style={{ display:"flex", gap:"1rem" }}>
              <button className="btn-outline" style={{ flex:1 }} onClick={() => { setStep(1); setResult(null); }}>Recalculate</button>
              <button className="btn-orange" style={{ flex:1 }} onClick={() => navigate('/dashboard')}>Save My Plan 💾</button>
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <footer style={{ background:"#1a2f45", color:"#94a3b8", padding:"1.5rem 2rem", marginTop:"2rem" }}>
        <div style={{ maxWidth:720, margin:"0 auto", display:"flex", justifyContent:"space-between", flexWrap:"wrap", gap:"1rem" }}>
          <div style={{ display:"flex", gap:"1.5rem" }}>
            { ["Home","Fitness","Interview","Performance"].map(l => <span key={l} style={{ fontSize: ".72rem", cursor:"pointer" }}>{l}</span>) }
          </div>
          <div style={{ display:"flex", gap:"1.5rem" }}>
            { ["FAQs","Feedback","App Download","Membership Plans"].map(l => <span key={l} style={{ fontSize: ".72rem", cursor:"pointer" }}>{l}</span>) }
          </div>
        </div>
      </footer>
    </div>
  );
}
