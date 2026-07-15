import React, { useRef, useEffect, useState, useCallback } from "react";
import axios from "axios";
import { motion, AnimatePresence } from "framer-motion";
import {
  FiHome, FiCamera, FiType, FiBarChart2, FiAlertTriangle,
  FiInfo, FiSettings, FiVolume2, FiVolumeX, FiTrash2,
  FiCopy, FiCheck, FiChevronLeft, FiChevronRight,
  FiCameraOff, FiActivity, FiWifi, FiWifiOff, FiSearch,
  FiZap, FiGlobe, FiLayers, FiTarget,
  FiShield, FiSliders
} from "react-icons/fi";
import { MdSignLanguage, MdSpeed } from "react-icons/md";
import {
  AreaChart, Area, BarChart, Bar, XAxis, YAxis,
  Tooltip, ResponsiveContainer
} from "recharts";
import "./App.css";
import { t, translatePrediction, LANGUAGES } from "./translations";

// ─── Config ───────────────────────────────────────────────────────────────────
const API        = process.env.REACT_APP_API_URL || "http://127.0.0.1:5000";
const WASM_CDN   = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_PATH = `${window.location.origin}/hand_landmarker.task`;
const SEQ_LEN    = 30;

const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17],
];

const NAV = [
  { id:"home",      icon:FiHome,          label:"Dashboard"      },
  { id:"detection", icon:FiCamera,        label:"Live Detection" },
  { id:"text2sign", icon:FiType,          label:"Text to Sign"   },
  { id:"analytics", icon:FiBarChart2,     label:"Analytics"      },
  { id:"emergency", icon:FiAlertTriangle, label:"Emergency"      },
  { id:"about",     icon:FiInfo,          label:"About"          },
  { id:"settings",  icon:FiSettings,      label:"Settings"       },
];

// ─── Pure helpers ─────────────────────────────────────────────────────────────
const fmtTime = (s) => {
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  return h > 0
    ? `${String(h).padStart(2,"0")}:${String(m).padStart(2,"0")}:${String(sec).padStart(2,"0")}`
    : `${String(m).padStart(2,"0")}:${String(sec).padStart(2,"0")}`;
};
const qualityLabel = (q) => q >= 85 ? "Excellent" : q >= 65 ? "Good" : q >= 40 ? "Fair" : "Poor";
const confLabel    = (c) => c >= 90 ? "High" : c >= 70 ? "Good" : c >= 50 ? "Medium" : "Low";
const latLabel     = (l) => l < 50 ? "Fast" : l < 120 ? "Good" : "Slow";
const fpsColor     = (f) => f >= 20 ? "var(--green)" : f >= 10 ? "var(--orange)" : "var(--pink)";
const getSignImg   = (c) => (c && /^[A-Z]$/.test(c))
  ? `https://www.lifeprint.com/asl101/fingerspelling/images/${c.toLowerCase()}.gif`
  : null;

const formatSentence = (words) => {
  let result = "";
  let prevWasChar = false;
  for (let i = 0; i < words.length; i++) {
    const w = words[i];
    if (w === " " || w === "") {
      result += " ";
      prevWasChar = false;
    } else if (w.length === 1) {
      if (prevWasChar) {
        result += w;
      } else {
        if (result.length > 0 && !result.endsWith(" ")) {
          result += " ";
        }
        result += w;
      }
      prevWasChar = true;
    } else {
      if (result.length > 0 && !result.endsWith(" ")) {
        result += " ";
      }
      result += w;
      prevWasChar = false;
    }
  }
  return result;
};

function drawSkeleton(ctx, handsLandmarks, W, H) {
  const TIPS = [4,8,12,16,20];
  const CLR  = ["#00d4ff","#a855f7"];
  const GLOW = ["rgba(0,212,255,0.55)","rgba(168,85,247,0.55)"];

  handsLandmarks.forEach((hand, hi) => {
    const c = CLR[hi%2], g = GLOW[hi%2];
    ctx.save();
    ctx.shadowColor = g; ctx.shadowBlur = 10;
    ctx.strokeStyle = c; ctx.lineWidth = 2.5; ctx.lineCap = "round";
    HAND_CONNECTIONS.forEach(([a,b]) => {
      if (!hand[a] || !hand[b]) return;
      ctx.beginPath();
      ctx.moveTo(hand[a].x*W, hand[a].y*H);
      ctx.lineTo(hand[b].x*W, hand[b].y*H);
      ctx.stroke();
    });
    ctx.restore();

    hand.forEach((lm, i) => {
      const x = lm.x*W, y = lm.y*H;
      const r = i===0 ? 9 : TIPS.includes(i) ? 7 : 5;
      ctx.save();
      ctx.beginPath(); ctx.arc(x,y,r+4,0,Math.PI*2);
      ctx.fillStyle = g.replace("0.55)","0.12)"); ctx.fill();
      ctx.shadowColor = g; ctx.shadowBlur = 14;
      ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2);
      ctx.fillStyle = i===0 ? "#fff" : c; ctx.fill();
      if (i===0 || TIPS.includes(i)) {
        ctx.shadowBlur = 0;
        ctx.beginPath(); ctx.arc(x,y,r*0.38,0,Math.PI*2);
        ctx.fillStyle = "rgba(255,255,255,0.9)"; ctx.fill();
      }
      ctx.restore();
    });
  });
  ctx.shadowBlur = 0;
}

// ─── CircularGauge ────────────────────────────────────────────────────────────
const CircularGauge = ({ value=0, size=130, color="#00d4ff", trackColor="rgba(255,255,255,0.06)" }) => {
  const r = (size-20)/2, circ = 2*Math.PI*r;
  const offset = circ - (Math.min(value,100)/100)*circ;
  const id = `g${Math.round(value)}${color.replace(/[^a-z0-9]/gi,"")}`;
  return (
    <svg width={size} height={size} style={{overflow:"visible"}}>
      <defs>
        <filter id={id}><feGaussianBlur stdDeviation="3" result="blur"/>
          <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
      </defs>
      <circle cx={size/2} cy={size/2} r={r} stroke={trackColor} strokeWidth={10} fill="none"/>
      <circle cx={size/2} cy={size/2} r={r} stroke={color} strokeWidth={10} fill="none"
        strokeDasharray={circ} strokeDashoffset={offset} strokeLinecap="round"
        transform={`rotate(-90 ${size/2} ${size/2})`}
        style={{transition:"stroke-dashoffset 0.45s ease", filter:`url(#${id})`}}/>
      <text x={size/2} y={size/2-2} textAnchor="middle" fill="#f1f5f9"
        fontSize={size*0.195} fontWeight="800" fontFamily="Space Grotesk, sans-serif">
        {Math.round(value)}%
      </text>
      <text x={size/2} y={size/2+size*0.135} textAnchor="middle"
        fill="rgba(255,255,255,0.35)" fontSize={size*0.09} fontFamily="Inter, sans-serif">
        CONFIDENCE
      </text>
    </svg>
  );
};

// ─── StatMini ─────────────────────────────────────────────────────────────────
const StatMini = ({ icon, value, unit="", label, sub, color="#00d4ff" }) => (
  <div className="stat-mini">
    <div className="sm-icon" style={{color}}>{icon}</div>
    <div className="sm-value" style={{color}}>{value}<span className="sm-unit">{unit}</span></div>
    <div className="sm-label">{label}</div>
    {sub && <div className="sm-sub" style={{color}}>{sub}</div>}
  </div>
);

// DetectionInfoPanel removed for production

// ─── Text2Sign Page ───────────────────────────────────────────────────────────
const Text2SignPage = ({ inputText,setInputText,idx,isAnim,speed,setSpeed,start,reset,speak,language }) => {
  const chars = inputText.toUpperCase().split("");
  const cur   = chars[idx];
  const img   = getSignImg(cur);
  return (
    <div className="page t2s-page">
      <h2 className="page-title">{t("Text to Sign Language", language)}</h2>
      <p className="page-sub">{t("Convert text into animated ASL sign gestures", language)}</p>
      <div className="t2s-layout">
        <div className="t2s-left">
          <div className="glass-card">
            <div className="card-hd"><FiType/> {t("Input Text", language)}</div>
            <textarea className="t2s-area" rows={5} placeholder={t("Type your message here...", language)}
              value={inputText} onChange={e=>setInputText(e.target.value)}/>
            <div className="t2s-btns">
              <button className="btn-primary" onClick={start} disabled={isAnim||!inputText.trim()}>▶ {t("Convert", language)}</button>
              <button className="btn-ghost" onClick={reset}>↺ {t("Reset", language)}</button>
              <button className="btn-ghost" onClick={()=>speak(inputText,true)}><FiVolume2/> {t("Speak", language)}</button>
            </div>
          </div>
          <div className="glass-card">
            <div className="card-hd">⚡ {t("Speed — ", language)}{speed}x</div>
            <div className="slider-row">
              <span>0.5x</span>
              <input type="range" min="0.5" max="2" step="0.1" value={speed}
                onChange={e=>setSpeed(parseFloat(e.target.value))} style={{flex:1,accentColor:"var(--cyan)"}}/>
              <span>2x</span>
            </div>
            <div className="chars-row">
              {chars.map((c,i) => (
                <span key={i} className={`cc${i===idx?" active":i<idx?" done":""}`}>
                  {c===" "?"⎵":c}
                </span>
              ))}
            </div>
          </div>
        </div>
        <div className="t2s-right">
          <div className="glass-card sign-card">
            <div className="card-hd"><MdSignLanguage/> {t("Sign Animation", language)}</div>
            <div className="sign-display">
              {idx>=0&&cur ? (
                <AnimatePresence mode="wait">
                  <motion.div key={idx} initial={{scale:0,rotate:-10}} animate={{scale:1,rotate:0}}
                    exit={{scale:0,rotate:10}} transition={{type:"spring",stiffness:280}}>
                    {img ? (
                      <div className="sign-img-wrap">
                        <img src={img} alt={cur} className="sign-img"
                          onError={e=>{e.target.style.display="none";e.target.nextSibling.style.display="flex";}}/>
                        <div className="sign-img-fb" style={{display:"none"}}>{cur}</div>
                        <p className="sign-lbl">{cur}</p>
                      </div>
                    ) : <span className="sign-space">⎵</span>}
                  </motion.div>
                </AnimatePresence>
              ) : (
                <div className="sign-ph"><MdSignLanguage/><p>{t("Enter text & click Convert", language)}</p></div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// ─── Home Page ────────────────────────────────────────────────────────────────
const HomePage = ({ setPage, analytics, language }) => (
  <div className="page home-page">
    <div className="home-hero">
      <motion.div className="hero-icon-wrap"
        animate={{boxShadow:["0 0 30px rgba(0,212,255,0.3)","0 0 60px rgba(0,212,255,0.6)","0 0 30px rgba(0,212,255,0.3)"]}}
        transition={{duration:2,repeat:Infinity}}>
        <MdSignLanguage className="hero-icon"/>
      </motion.div>
      <motion.h1 className="hero-title" initial={{opacity:0,y:30}} animate={{opacity:1,y:0}} transition={{delay:0.15}}>
        {t("AI Sign Language", language)}<br/><span className="hero-grad">{t("Communication System", language)}</span>
      </motion.h1>
      <motion.p className="hero-sub" initial={{opacity:0}} animate={{opacity:1}} transition={{delay:0.3}}>
        {t("Real-time ASL recognition powered by MediaPipe & deep learning. Bridging silence, building connections.", language)}
      </motion.p>
      <motion.div className="hero-btns" initial={{opacity:0,y:20}} animate={{opacity:1,y:0}} transition={{delay:0.45}}>
        <button className="btn-primary btn-lg" onClick={()=>setPage("detection")}>
          <FiCamera/> {t("Start Detection →", language)}
        </button>
        <button className="btn-ghost btn-lg" onClick={()=>setPage("text2sign")}>
          <FiType/> {t("Text to Sign", language)}
        </button>
      </motion.div>
    </div>
    <div className="home-stats">
      {[
        {val:"26",  lbl:t("Alphabet Signs", language),   icon:"🔤", c:"cyan"  },
        {val:"25",  lbl:t("WLASL Word Signs", language),  icon:"💬", c:"purple"},
        {val:"20+", lbl:t("Target FPS", language),        icon:"⚡", c:"green" },
        {val:"9",   lbl:t("Languages", language),         icon:"🌐", c:"blue"  },
      ].map((s,i) => (
        <motion.div key={i} className={`home-stat-card ${s.c}`}
          initial={{opacity:0,y:20}} animate={{opacity:1,y:0}} transition={{delay:0.6+i*0.1}}>
          <span className="hsc-icon">{s.icon}</span>
          <p className="hsc-val">{s.val}</p>
          <p className="hsc-lbl">{s.lbl}</p>
        </motion.div>
      ))}
    </div>
    <div className="home-features">
      {[
        {icon:<FiZap/>,   title:t("Browser MediaPipe", language),     desc:t("Landmark detection runs in your browser via WebGL — no server round-trip for vision", language)},
        {icon:<FiTarget/>,title:t("Dual AI Models", language),        desc:t("Random Forest for alphabets + LSTM neural network for WLASL word signs", language)},
        {icon:<FiGlobe/>, title:t("Multi-Language TTS", language),    desc:t("Text-to-speech in 9 languages", language)},
        {icon:<FiShield/>,title:t("Real-Time Analytics", language),   desc:t("Session tracking, confidence trending, gesture frequency analytics", language)},
      ].map((f,i) => (
        <motion.div key={i} className="feat-card"
          initial={{opacity:0,y:16}} animate={{opacity:1,y:0}} transition={{delay:0.9+i*0.1}}
          whileHover={{y:-4,transition:{duration:0.2}}}>
          <div className="feat-icon">{f.icon}</div>
          <h3 className="feat-title">{f.title}</h3>
          <p className="feat-desc">{f.desc}</p>
        </motion.div>
      ))}
    </div>
  </div>
);

// ─── Analytics Page ───────────────────────────────────────────────────────────
const AnalyticsPage = ({ analytics, fmtSessTime, language }) => {
  const freqData = Object.entries(analytics.gestureFreq)
    .sort((a,b)=>b[1]-a[1]).slice(0,8).map(([n,c])=>({name:translatePrediction(n, language),count:c}));
  return (
    <div className="page analytics-page">
      <h2 className="page-title">{t("Analytics", language)}</h2>
      <p className="page-sub">{t("Session performance metrics and gesture frequency", language)}</p>
      <div className="analytics-grid">
        {[
          {v:analytics.totalGestures, l:t("Total Gestures", language),   icon:"🤟",c:"cyan"  },
          {v:`${analytics.avgConfidence}%`,l:t("Avg Confidence", language),icon:"🎯",c:"green" },
          {v:fmtSessTime(analytics.sessionTime||0),l:t("Session Time", language),icon:"⏱️",c:"blue"  },
          {v:analytics.sessions,      l:t("Sessions", language),          icon:"📊",c:"purple"},
        ].map((s,i) => (
          <div key={i} className={`an-stat ${s.c}`}>
            <span className="an-icon">{s.icon}</span>
            <p className="an-val">{s.v}</p>
            <p className="an-lbl">{s.l}</p>
          </div>
        ))}
      </div>
      <div className="charts-grid">
        <div className="glass-card chart-card">
          <h3>📈 {t("Confidence Trend", language)}</h3>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={analytics.confidenceTrend}>
              <defs>
                <linearGradient id="cg" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor="#00d4ff" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#00d4ff" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <XAxis dataKey="time" tick={{fill:"#64748b",fontSize:10}}/>
              <YAxis domain={[0,100]} tick={{fill:"#64748b",fontSize:10}}/>
              <Tooltip contentStyle={{background:"#0d1225",border:"1px solid rgba(0,212,255,0.2)",borderRadius:12}}/>
              <Area type="monotone" dataKey="confidence" stroke="#00d4ff" fill="url(#cg)" strokeWidth={2}/>
            </AreaChart>
          </ResponsiveContainer>
        </div>
        <div className="glass-card chart-card">
          <h3>📊 {t("Gesture Frequency", language)}</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={freqData}>
              <XAxis dataKey="name" tick={{fill:"#64748b",fontSize:10}}/>
              <YAxis tick={{fill:"#64748b",fontSize:10}}/>
              <Tooltip contentStyle={{background:"#0d1225",border:"1px solid rgba(168,85,247,0.2)",borderRadius:12}}/>
              <Bar dataKey="count" fill="#a855f7" radius={[4,4,0,0]}/>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

// ─── Emergency Page ───────────────────────────────────────────────────────────
const EmergencyPage = ({ settings, setSettings, speak, language }) => (
  <div className="page emergency-page">
    <div className="em-hero">
      <motion.div className="em-icon-wrap"
        animate={{boxShadow:["0 0 20px rgba(244,63,94,0.5)","0 0 50px rgba(244,63,94,0.8)","0 0 20px rgba(244,63,94,0.5)"]}}
        transition={{duration:1.5,repeat:Infinity}}>
        <FiAlertTriangle/>
      </motion.div>
      <h2 className="em-title">{t("Emergency Assistance", language)}</h2>
      <p className="em-sub">{t("One-tap emergency communication", language)}</p>
      <button className={`sound-btn ${settings.voiceOutput?"on":"off"}`}
        onClick={()=>setSettings(s=>({...s,voiceOutput:!s.voiceOutput}))}>
        {settings.voiceOutput?<FiVolume2/>:<FiVolumeX/>} {settings.voiceOutput ? t("Sound On", language) : t("Sound Off", language)}
      </button>
    </div>
    <div className="em-grid">
      {[
        {l:t("I Need Help!", language),        m:t("I need help! Please assist me immediately.", language),        icon:"🤝",c:"cyan"  },
        {l:t("Call My Family!", language),     m:t("Please call my family member immediately.", language),          icon:"📞",c:"orange"},
        {l:t("Medical Emergency!", language),  m:t("Medical emergency! I need immediate attention!", language),    icon:"🚨",c:"pink"  },
        {l:t("Call Police!", language),        m:t("Please call the police immediately!", language),               icon:"🚔",c:"blue"  },
        {l:t("I Am Lost!", language),          m:t("I am lost. Please help me find my way.", language),            icon:"📍",c:"green" },
        {l:t("I Can't Speak!", language),      m:t("I cannot speak. Please be patient with me.", language),       icon:"🤐",c:"purple"},
      ].map((e,i) => (
        <motion.button key={i} className={`em-card ${e.c}`} whileTap={{scale:0.96}}
          onClick={()=>{speak(e.m,true);alert(e.m);}}>
          <span className="em-card-icon">{e.icon}</span>
          <p className="em-card-title">{e.l}</p>
          <p className="em-card-msg">{e.m}</p>
        </motion.button>
      ))}
    </div>
  </div>
);

// ─── About Page ───────────────────────────────────────────────────────────────
const AboutPage = ({ language }) => (
  <div className="page about-page">
    <h2 className="page-title">{t("About SilentSpeak", language)}</h2>
    <p className="page-sub">{t("Breaking barriers with AI-powered sign language recognition", language)}</p>
    <div className="about-grid">
      <div className="glass-card"><h3>🎯 {t("Mission", language)}</h3>
        <p>{t("SilentSpeak bridges the communication gap between deaf and hearing communities using real-time AI sign language recognition. Our goal is accessible, instant communication for everyone.", language)}</p>
      </div>
      <div className="glass-card"><h3>⚙️ {t("Technology Stack", language)}</h3>
        <ul className="about-list">
          <li>🧠 {t("Random Forest — ASL alphabet (A–Z)", language)}</li>
          <li>🔄 {t("LSTM Neural Network — WLASL word signs", language)}</li>
          <li>👁️ {t("MediaPipe HandLandmarker (browser-side WebGL)", language)}</li>
          <li>⚛️ React 18 + Flask + TensorFlow</li>
          <li>🌐 {t("Multi-language TTS (9 languages)", language)}</li>
        </ul>
      </div>
      <div className="glass-card"><h3>📊 {t("Performance", language)}</h3>
        <div className="acc-bars">
          {[
            {l:"Browser MediaPipe (WebGL)", v:95},
            {l:t("Alphabet Recognition", language),      v:95},
            {l:t("WLASL Word Recognition", language),    v:94},
          ].map((a,i) => (
            <div key={i} className="acc-row">
              <div className="acc-hd"><span>{a.l}</span><span className="acc-pct">{a.v}%</span></div>
              <div className="acc-track">
                <motion.div className="acc-fill" initial={{width:0}}
                  animate={{width:`${a.v}%`}} transition={{delay:i*0.2,duration:1}}/>
              </div>
            </div>
          ))}
        </div>
      </div>
      <div className="glass-card"><h3>📦 {t("Datasets", language)}</h3>
        <ul className="about-list">
          <li>📘 ASL Alphabet (Kaggle) — 87,000 images</li>
          <li>📗 WLASL Dataset — 11,980 word sign videos</li>
          <li>🔬 Data augmentation — 5× multiplication</li>
          <li>✅ {t("25 WLASL word classes trained", language)}</li>
        </ul>
      </div>
    </div>
  </div>
);

// ─── Settings Page ────────────────────────────────────────────────────────────
const SettingsPage = ({ language, setLanguage, settings, setSettings }) => (
  <div className="page settings-page">
    <h2 className="page-title">{t("Settings", language)}</h2>
    <p className="page-sub">{t("Customize your SilentSpeak experience", language)}</p>
    <div className="settings-col">
      <div className="glass-card">
        <h3><FiGlobe/> {t("Language", language)}</h3>
        <div className="lang-grid">
          {LANGUAGES.map(l => (
            <button key={l.code} className={`lang-tile ${language===l.code?"active":""}`}
              onClick={()=>setLanguage(l.code)}>
              <p className="lt-main">{l.label}</p>
              <p className="lt-sub">{l.native}</p>
            </button>
          ))}
        </div>
      </div>

      <div className="glass-card">
        <h3><FiVolume2/> {t("Voice Output", language)}</h3>
        <div className="setting-row">
          <span>{t("Enable Voice", language)}</span>
          <div className={`toggle ${settings.voiceOutput?"on":""}`}
            onClick={()=>setSettings(s=>({...s,voiceOutput:!s.voiceOutput}))}>
            <div className="toggle-knob"/>
          </div>
        </div>
        {[
          {k:"speechRate",l:t("Speech Rate", language),min:0.5,max:2,step:0.1,fmt:v=>`${v}x`},
          {k:"pitch",     l:t("Pitch", language),      min:0.5,max:2,step:0.1,fmt:v=>`${v}`},
          {k:"volume",    l:t("Volume", language),     min:0,  max:1,step:0.1,fmt:v=>`${Math.round(v*100)}%`},
        ].map(s => (
          <div key={s.k} className="setting-slider">
            <div className="ssl-hd"><span>{s.l}</span><span className="ssl-v">{s.fmt(settings[s.k])}</span></div>
            <input type="range" min={s.min} max={s.max} step={s.step} value={settings[s.k]}
              onChange={e=>setSettings(p=>({...p,[s.k]:parseFloat(e.target.value)}))}
              style={{accentColor:"var(--cyan)"}}/>
          </div>
        ))}
      </div>

      <div className="glass-card">
        <h3><FiSliders/> {t("Detection", language)}</h3>
        {[
          {k:"minConfAlpha",l:t("Min Alphabet Confidence", language),min:10,max:99,step:1,fmt:v=>`${v}%`},
          {k:"minConfWord", l:t("Min Word Confidence", language),    min:30,max:99,step:1,fmt:v=>`${v}%`},
          {k:"holdFrames",  l:t("Hold Frames", language),            min:2, max:10,step:1,fmt:v=>`${v} fr`},
        ].map(s => (
          <div key={s.k} className="setting-slider">
            <div className="ssl-hd"><span>{s.l}</span><span className="ssl-v">{s.fmt(settings[s.k])}</span></div>
            <input type="range" min={s.min} max={s.max} step={s.step} value={settings[s.k]}
              onChange={e=>setSettings(p=>({...p,[s.k]:parseFloat(e.target.value)}))}
              style={{accentColor:"var(--cyan)"}}/>
          </div>
        ))}
        <div className="setting-row">
          <span>{t("Show Skeleton Overlay", language)}</span>
          <div className={`toggle ${settings.showSkeleton?"on":""}`}
            onClick={()=>setSettings(s=>({...s,showSkeleton:!s.showSkeleton}))}>
            <div className="toggle-knob"/>
          </div>
        </div>
        <div className="setting-row">
          <span>{t("High Contrast Mode", language)}</span>
          <div className={`toggle ${settings.highContrast?"on":""}`}
            onClick={()=>setSettings(s=>({...s,highContrast:!s.highContrast}))}>
            <div className="toggle-knob"/>
          </div>
        </div>
      </div>

      <button className="reset-btn" onClick={()=>setSettings({
        voiceOutput:true,speechRate:1,pitch:1,volume:0.8,
        minConfAlpha:15,minConfWord:35,holdFrames:12,
        showSkeleton:true,highContrast:false,
      })}>↺ {t("Reset to Defaults", language)}</button>
    </div>
  </div>
);

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  // DOM refs
  const videoRef    = useRef(null);
  const canvasRef   = useRef(null);
  // MediaPipe
  const handRef     = useRef(null);
  // Loop refs
  const animRef     = useRef(null);
  const inFlightRef = useRef(false);
  const lastCallTimeRef = useRef(0);
  const fpsRef      = useRef({count:0,last:Date.now()});
  // Detection state refs
  const seqBufRef   = useRef([]);
  const lastVecRef  = useRef(null);
  const prevSignRef = useRef({sign:"",count:0});
  const lastAddRef  = useRef({sign:"",time:0});
  const sessionRef  = useRef(null);
  const zeroHandFramesRef = useRef(0);
  // Stable callbacks
  const settingsRef = useRef({});
  const speakRef    = useRef(null);
  const predHandlerRef = useRef(null);
  
  // Throttling refs for FPS optimization
  const handCountRef = useRef(0);
  const seqCountRef  = useRef(0);
  const frameCounterRef = useRef(0);

  // ── Global state ──────────────────────────────────────────────────────────
  const [page,       setPage]       = useState("home");
  const [sidebar,    setSidebar]    = useState(true);
  const [apiStatus,  setApiStatus]  = useState("checking");
  const [mpStatus,   setMpStatus]   = useState("loading"); // loading|ready|error
  const [detectionMode, setDetectionMode] = useState("alphabet");

  // Camera
  const [cameraOn,    setCameraOn]    = useState(false);
  const [camStream,   setCamStream]   = useState(null);
  const [devices,     setDevices]     = useState([]);
  const [selDevice,   setSelDevice]   = useState("");

  // Detection
  const [handCount,  setHandCount]  = useState(0);
  const [currentSign,setCurrentSign]= useState("");
  const [currentWord,setCurrentWord]= useState("");
  const [alphaConf,  setAlphaConf]  = useState(0);
  const [wordConf,   setWordConf]   = useState(0);
  const [fps,        setFps]        = useState(0);
  const [latency,    setLatency]    = useState(0);
  const [seqCount,   setSeqCount]   = useState(0);
  const [stability,  setStability]  = useState(0);
  const [sessTime,   setSessTime]   = useState(0);
  const [rawAlpha,   setRawAlpha]   = useState("—");
  const [rawWord,    setRawWord]    = useState("—");
  const [rawAlphaConf, setRawAlphaConf] = useState(0);
  const [rawWordConf,  setRawWordConf]  = useState(0);
  const [activeModel,  setActiveModel]  = useState("None");
  const [finalPrediction, setFinalPrediction] = useState("—");
  const [predHistory,setPredHistory]= useState([]);
  const [sentence,   setSentence]   = useState([]);
  const [speaking,   setSpeaking]   = useState(false);
  const [copied,     setCopied]     = useState(false);
  const [language,   setLanguage]   = useState("en");

  // Versions and Debug States
  const [backendVersion, setBackendVersion] = useState("");
  const [modelVersion, setModelVersion] = useState("");
  const [labelMapVersion, setLabelMapVersion] = useState("");
  const [rejectionReason, setRejectionReason] = useState("");

  // Hands-free Automatic Sign Detection States
  const [isCapturing, setIsCapturing] = useState(false);
  const isCapturingRef = useRef(false);
  const [cooldownActive, setCooldownActive] = useState(false);
  const cooldownActiveRef = useRef(false);
  const motionHistoryRef = useRef([]);
  const lastHandednessRef = useRef("Right");

  const [settings, setSettings] = useState({
    voiceOutput:true,speechRate:1,pitch:1,volume:0.8,
    minConfAlpha:15,minConfWord:35,holdFrames:12,
    showSkeleton:true,highContrast:false,
  });

  const [analytics, setAnalytics] = useState({
    totalGestures:0,avgConfidence:0,sessions:0,sessionTime:0,
    gestureFreq:{},confidenceTrend:[],
  });

  // Text-to-Sign
  const [t2sText,    setT2sText]    = useState("");
  const [t2sIdx,     setT2sIdx]     = useState(-1);
  const [t2sAnim,    setT2sAnim]    = useState(false);
  const [t2sSpeed,   setT2sSpeed]   = useState(1);
  const t2sRef = useRef(null);

  // Sync settings ref
  useEffect(()=>{ settingsRef.current = settings; }, [settings]);

  // ── Load MediaPipe ─────────────────────────────────────────────────────────
  useEffect(()=>{
    let cancelled = false;
    (async ()=>{
      try {
        const { HandLandmarker, FilesetResolver } = await import("@mediapipe/tasks-vision");
        const fs = await FilesetResolver.forVisionTasks(WASM_CDN);
        const opts = {
          baseOptions:{ modelAssetPath:MODEL_PATH, delegate:"GPU" },
          runningMode:"VIDEO", numHands:2,
          minHandDetectionConfidence:0.55,
          minHandPresenceConfidence:0.55,
          minTrackingConfidence:0.55,
        };
        let hl;
        try { hl = await HandLandmarker.createFromOptions(fs, opts); }
        catch {
          hl = await HandLandmarker.createFromOptions(fs, {...opts, baseOptions:{...opts.baseOptions,delegate:"CPU"}});
        }
        if (!cancelled) { handRef.current = hl; setMpStatus("ready"); }
      } catch(e) {
        console.error("MediaPipe load failed:", e);
        if (!cancelled) setMpStatus("error");
      }
    })();
    return ()=>{ cancelled=true; };
  },[]);

  // ── API health ─────────────────────────────────────────────────────────────
  useEffect(()=>{
    const chk = () => axios.get(`${API}/health`)
      .then(res => {
        setApiStatus("online");
        if (res.data) {
          setBackendVersion(res.data.backend_version || "Unknown");
          setModelVersion(res.data.model_version || "Unknown");
          setLabelMapVersion(res.data.label_map_version || "Unknown");
        }
      })
      .catch(() => {
        setApiStatus("offline");
      });
    chk(); const t=setInterval(chk,10000); return ()=>clearInterval(t);
  },[]);

  // ── Session timer ──────────────────────────────────────────────────────────
  useEffect(()=>{
    if (cameraOn) {
      sessionRef.current = Date.now();
      setAnalytics(a=>({...a,sessions:a.sessions+1}));
      const t=setInterval(()=>{
        setSessTime(Math.floor((Date.now()-sessionRef.current)/1000));
        setAnalytics(a=>({...a,sessionTime:Math.floor((Date.now()-sessionRef.current)/1000)}));
      },1000);
      return ()=>clearInterval(t);
    }
  },[cameraOn]);

  // ── Enumerate cameras ──────────────────────────────────────────────────────
  useEffect(()=>{
    navigator.mediaDevices?.enumerateDevices().then(all=>{
      const vids=all.filter(d=>d.kind==="videoinput");
      setDevices(vids);
      if(vids.length&&!selDevice) setSelDevice(vids[0].deviceId);
    }).catch(()=>{});
  // eslint-disable-next-line
  },[]);

  // ── Start / Stop camera ────────────────────────────────────────────────────
  const startCamera = useCallback(async (devId=null)=>{
    const target = devId||selDevice;
    const candidates = target
      ? [
          {video:{deviceId:{exact:target},width:640,height:480}},
          {video:{deviceId:{exact:target}}},
          {video:{width:640,height:480,facingMode:"user"}},
          {video:true}
        ]
      : [
          {video:{width:640,height:480,facingMode:"user"}},
          {video:true}
        ];
    let stream=null;
    let errorMsg = "Camera access failed. Please ensure camera permissions are granted and no other application is using it.";
    for(const c of candidates){
      try{
        stream=await navigator.mediaDevices.getUserMedia(c);
        break;
      }catch(err){
        console.warn("Camera attempt failed for constraint:", c, err);
        if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
          errorMsg = "Camera access denied. Please click the camera icon in your browser address bar to allow permissions.";
        } else if (err.name === "NotReadableError" || err.name === "TrackStartError") {
          errorMsg = "Webcam is already in use by another application (like Zoom, Teams, or another tab). Please close it and retry.";
        } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
          errorMsg = "No webcam device detected on your system.";
        }
      }
    }
    if(!stream){
      alert(errorMsg);
      return;
    }
    setCamStream(stream);
    setCameraOn(true);
    const all = await navigator.mediaDevices.enumerateDevices();
    setDevices(all.filter(d=>d.kind==="videoinput"));
    const t=stream.getVideoTracks()[0];
    if(t?.getSettings?.()?.deviceId) setSelDevice(t.getSettings().deviceId);
  },[selDevice]);

  const stopCamera = useCallback(()=>{
    if(animRef.current) cancelAnimationFrame(animRef.current);
    if(camStream) camStream.getTracks().forEach(t=>t.stop());
    if(videoRef.current) videoRef.current.srcObject=null;
    setCamStream(null); setCameraOn(false);
    setHandCount(0); setCurrentSign(""); setCurrentWord("");
    setAlphaConf(0); setWordConf(0); setFps(0); setLatency(0);
    setSeqCount(0); setStability(0); setSessTime(0);
    setRawAlpha("—"); setRawWord("—");
    setRawAlphaConf(0); setRawWordConf(0);
    setActiveModel("None"); setFinalPrediction("—");
    seqBufRef.current=[]; lastVecRef.current=null;
    prevSignRef.current={sign:"",count:0};
    inFlightRef.current=false;
  },[camStream]);

  const handleCamChange = useCallback(async (id)=>{
    setSelDevice(id);
    if(cameraOn){ await stopCamera(); await startCamera(id); }
  },[cameraOn,stopCamera,startCamera]);

  // Sync stream to video
  useEffect(()=>{
    if(videoRef.current) videoRef.current.srcObject=camStream;
  },[camStream]);

  const predictSequenceDirect = (seq, handLabel) => {
    if (seq.length < 5) return;
    
    setCooldownActive(true);
    cooldownActiveRef.current = true;
    
    const payload = { 
      sequence: seq,
      handedness: handLabel || "Right"
    };
    
    axios.post(`${API}/predict_word`, payload, { timeout: 3000 })
      .then(res => {
        const pred = res.data.prediction;
        const conf = res.data.confidence;
        
        console.log(`[Auto Word Prediction] Result: ${pred} (${conf}%)`);
        
        if (pred) {
          setCurrentWord(pred);
          setWordConf(conf);
          setFinalPrediction(pred);
          setRejectionReason("");
          
          const now = Date.now();
          setPredHistory(prev => {
            if (prev[0]?.sign === pred && now - prev[0]?.id < 1500) return prev;
            return [{ sign: pred, isWord: true, conf: conf, id: now }, ...prev].slice(0, 14);
          });
          
          setSentence(p => [...p, pred]);
          
          if (settingsRef.current.voiceOutput && speakRef.current) {
            speakRef.current(pred);
          }
          
          setAnalytics(a => {
            const tot = a.totalGestures + 1;
            return {
              ...a,
              totalGestures: tot,
              avgConfidence: Math.round((a.avgConfidence * (tot - 1) + conf) / tot),
              gestureFreq: { ...a.gestureFreq, [pred]: (a.gestureFreq[pred] || 0) + 1 },
              confidenceTrend: [
                ...a.confidenceTrend.slice(-19),
                {
                  time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
                  confidence: Math.round(conf)
                }
              ]
            };
          });
        } else {
          setRejectionReason(res.data.rejection_reason || "Low confidence");
        }
      })
      .catch((err) => {
        console.error("Prediction API failed", err);
      })
      .finally(() => {
        setTimeout(() => {
          setCooldownActive(false);
          cooldownActiveRef.current = false;
          seqBufRef.current = [];
          setSeqCount(0);
          setIsCapturing(false);
          isCapturingRef.current = false;
          motionHistoryRef.current = [];
        }, 1500);
      });
  };

  // ── Prediction handler (ref-based — always fresh) ─────────────────────────
  predHandlerRef.current = (data)=>{
    const { alpha_sign:as, alpha_confidence:ac, word_sign:ws, word_confidence:wc } = data;

    // 1. Update raw debug states using correct raw confidence keys
    setRawAlpha(data.raw_alpha || "—");
    setRawAlphaConf(data.raw_alpha_conf || 0);
    setRawWord(data.raw_word || "—");
    setRawWordConf(data.raw_word_conf || 0);

    const activeM = data.active_model || "None";
    setActiveModel(activeM);
    setFinalPrediction(data.final_prediction || "—");
    setRejectionReason(data.rejection_reason || "");

    // 3. Update active display panels based on detectionMode
    if (detectionMode === "alphabet") {
      if (as) {
        setCurrentSign(as);
        setAlphaConf(ac);
      } else {
        setCurrentSign("");
        setAlphaConf(0);
      }
      setCurrentWord("");
      setWordConf(0);
    } else {
      if (ws) {
        setCurrentWord(ws);
        setWordConf(wc);
        seqBufRef.current = [];
        setSeqCount(0);
      }
      setCurrentSign("");
      setAlphaConf(0);
    }

    // 4. Update Sentence and Prediction History
    // Case A: Word predicted (instant trigger, no hold frames)
    if (detectionMode === "word" && ws) {
      const now = Date.now();
      setPredHistory(prev => {
        if (prev[0]?.sign === ws && now - prev[0]?.id < 1500) return prev;
        return [{ sign: ws, isWord: true, conf: wc, id: now }, ...prev].slice(0, 14);
      });

      if (ws !== lastAddRef.current.sign || now - lastAddRef.current.time > 1500) {
        setSentence(p => [...p, ws]);
        lastAddRef.current = { sign: ws, time: now };
        if (settingsRef.current.voiceOutput && speakRef.current) speakRef.current(ws);

        setAnalytics(a => {
          const tot = a.totalGestures + 1;
          return {
            ...a,
            totalGestures: tot,
            avgConfidence: Math.round((a.avgConfidence * (tot - 1) + wc) / tot),
            gestureFreq: { ...a.gestureFreq, [ws]: (a.gestureFreq[ws] || 0) + 1 },
            confidenceTrend: [
              ...a.confidenceTrend.slice(-19),
              {
                time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
                confidence: Math.round(wc)
              }
            ]
          };
        });
      }
    }
    // Case B: Alphabet predicted (requires hold frames stabilization)
    else if (detectionMode === "alphabet" && as) {
      const ref = prevSignRef.current;
      if (as === ref.sign) {
        ref.count++;
      } else {
        ref.sign = as;
        ref.count = 1;
      }

      const MIN = settingsRef.current.holdFrames || 3;
      if (ref.count >= MIN) {
        seqBufRef.current = [];
        setSeqCount(0);
      }

      if (ref.count === MIN) {
        const now = Date.now();
        setPredHistory(prev => {
          if (prev[0]?.sign === as && now - prev[0]?.id < 1500) return prev;
          return [{ sign: as, isWord: false, conf: ac, id: now }, ...prev].slice(0, 14);
        });

        const thresh = settingsRef.current.minConfAlpha || 15;
        if (ac >= thresh) {
          if (as !== lastAddRef.current.sign || now - lastAddRef.current.time > 1500) {
            setSentence(p => [...p, as]);
            lastAddRef.current = { sign: as, time: now };
            if (settingsRef.current.voiceOutput && speakRef.current) speakRef.current(as);

            setAnalytics(a => {
              const tot = a.totalGestures + 1;
              return {
                ...a,
                totalGestures: tot,
                avgConfidence: Math.round((a.avgConfidence * (tot - 1) + ac) / tot),
                gestureFreq: { ...a.gestureFreq, [as]: (a.gestureFreq[as] || 0) + 1 },
                confidenceTrend: [
                  ...a.confidenceTrend.slice(-19),
                  {
                    time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
                    confidence: Math.round(ac)
                  }
                ]
              };
            });
          }
        }
      }
    } else {
      prevSignRef.current = { sign: "", count: 0 };
    }
  };

  // ── Detection loop (rAF) ────────────────────────────────────────────────────
  useEffect(()=>{
    if(!cameraOn||!handRef.current||mpStatus!=="ready") return;
    let active=true;

    const detect=()=>{
      if(!active) return;
      frameCounterRef.current++;
      const video=videoRef.current;
      if(!video?.srcObject||video.readyState<2){
        animRef.current=requestAnimationFrame(detect); return;
      }

      // FPS counter
      fpsRef.current.count++;
      const now=Date.now();
      if(now-fpsRef.current.last>=1000){
        setFps(fpsRef.current.count);
        fpsRef.current={count:0,last:now};
      }

      // MediaPipe inference (GPU, sync)
      let results;
      try{ results=handRef.current.detectForVideo(video,performance.now()); }
      catch{ animRef.current=requestAnimationFrame(detect); return; }

      const nHands=results?.landmarks?.length||0;
      if (nHands !== handCountRef.current) {
        setHandCount(nHands);
        handCountRef.current = nHands;
      }

      // Canvas skeleton
      if(canvasRef.current && video.videoWidth>0){
        const cv=canvasRef.current;
        if(cv.width!==video.videoWidth){ cv.width=video.videoWidth; cv.height=video.videoHeight; }
        const ctx=cv.getContext("2d");
        ctx.clearRect(0,0,cv.width,cv.height);
        if(nHands>0 && settingsRef.current.showSkeleton){
          drawSkeleton(ctx,results.landmarks,cv.width,cv.height);
        }
      }

      if(nHands>0){
        zeroHandFramesRef.current = 0;

        // --- Hand Tracking Index Stabilization ---
        let primaryHandIndex = 0;
        if (results.handedness && results.handedness.length > 0) {
          const rightIdx = results.handedness.findIndex(h => h[0] && h[0].categoryName.toLowerCase() === "right");
          if (rightIdx !== -1) {
            primaryHandIndex = rightIdx;
          }
        }
        const hand = results.landmarks[primaryHandIndex];

        // Send raw landmarks from MediaPipe (server will handle handedness mapping)
        const rawVec = hand.flatMap(lm => [lm.x, lm.y, lm.z]);

        // Track handedness
        let handedness = "Right";
        if (results.handedness && results.handedness[primaryHandIndex] && results.handedness[primaryHandIndex][0]) {
          handedness = results.handedness[primaryHandIndex][0].categoryName;
        }
        lastHandednessRef.current = handedness;

        // Hands-Free Automatic capturing logic
        if (detectionMode === "word") {
          if (!cooldownActiveRef.current) {
            if (!isCapturingRef.current) {
              isCapturingRef.current = true;
              setIsCapturing(true);
              seqBufRef.current = [];
              motionHistoryRef.current = [];
              // Clear previous prediction to show we are capturing
              setCurrentWord("");
              setWordConf(0);
              setFinalPrediction("—");
            }
            
            seqBufRef.current.push(rawVec);
            if (seqBufRef.current.length > SEQ_LEN) {
              seqBufRef.current.shift();
            }
            
            // Track motion
            let motion = 0;
            if (lastVecRef.current) {
              motion = rawVec.reduce((sum, val, idx) => sum + Math.abs(val - lastVecRef.current[idx]), 0);
            }
            motionHistoryRef.current.push(motion);
            if (motionHistoryRef.current.length > 5) {
              motionHistoryRef.current.shift();
            }
            
            const currentLen = seqBufRef.current.length;
            if (currentLen % 5 === 0 || currentLen === SEQ_LEN || currentLen === 0) {
              setSeqCount(currentLen);
            }
            
            // Trigger immediately when sequence is complete (reaches 30 frames)
            if (seqBufRef.current.length === SEQ_LEN) {
              console.log(`[Auto Trigger] Sequence complete (${seqBufRef.current.length} frames). Predicting...`);
              const currentSeq = [...seqBufRef.current];
              setCooldownActive(true);
              cooldownActiveRef.current = true;
              predictSequenceDirect(currentSeq, handedness);
            }
          }
        }

        const ALPHA = 0.65;
        const smoothed = lastVecRef.current
          ? rawVec.map((v, i) => ALPHA * v + (1 - ALPHA) * lastVecRef.current[i])
          : rawVec;
        lastVecRef.current = smoothed;

        // Sequence buffer (only collect if in word mode!)
        if (detectionMode === "word") {
          // Automatic mode accumulates above, keep this here for visual/compatibility
        } else {
          seqBufRef.current = [];
        }

        // Stability (EMA delta, inverted to quality %) - throttled to 1 in 5 frames
        if (frameCounterRef.current % 5 === 0) {
          const delta = smoothed.reduce((acc, v, i) => acc + Math.abs(v - rawVec[i]), 0);
          setStability(Math.min(100, Math.round(100 - delta * 180)));
        }

        // Backend call (non-blocking, throttled to 15 FPS) - ONLY for alphabet mode
        const nowTime = performance.now();
        if (detectionMode === "alphabet" && !inFlightRef.current && (nowTime - lastCallTimeRef.current >= 66)) {
          inFlightRef.current = true;
          lastCallTimeRef.current = nowTime;
          const t0 = performance.now();

          const payload = {
            landmarks: hand.map(lm => [lm.x, lm.y, lm.z]),
            sequence: [],
            seq_len: 0,
            mode: "alphabet",
            min_conf_alpha: settingsRef.current.minConfAlpha,
            min_conf_word: settingsRef.current.minConfWord,
            handedness: handedness
          };

          axios.post(`${API}/predict_from_landmarks`, payload, { timeout: 3000 })
            .then(res => {
              if (frameCounterRef.current % 10 === 0) {
                setLatency(Math.round(performance.now() - t0));
              }
              if (predHandlerRef.current) predHandlerRef.current(res.data);
            })
            .catch(() => {})
            .finally(() => { inFlightRef.current = false; });
        }
      } else {
        zeroHandFramesRef.current++;
        setStability(0);

        // If we were capturing and lost hand, trigger prediction immediately
        if (detectionMode === "word" && isCapturingRef.current && !cooldownActiveRef.current) {
          const seq = seqBufRef.current;
          isCapturingRef.current = false;
          setIsCapturing(false);
          if (seq.length >= 15) {
            console.log(`[Auto Trigger] Hand lost. Triggering prediction with ${seq.length} frames...`);
            const currentSeq = [...seq];
            setCooldownActive(true);
            cooldownActiveRef.current = true;
            predictSequenceDirect(currentSeq, lastHandednessRef.current);
          } else {
            console.log(`[Auto Trigger] Hand lost, but sequence too short (${seq.length} frames). Resetting.`);
            seqBufRef.current = [];
            setSeqCount(0);
            motionHistoryRef.current = [];
          }
        }

        // Clear predictions after 10 consecutive zero-hand frames (grace period expired)
        if(zeroHandFramesRef.current > 10){
          if (!cooldownActiveRef.current) {
            setCurrentSign("");
            setCurrentWord("");
            setAlphaConf(0);
            setWordConf(0);
            setFinalPrediction("—");
          }
          setSeqCount(0);
          seqBufRef.current = [];
          lastVecRef.current = null;
        }
      }

      animRef.current=requestAnimationFrame(detect);
    };

    animRef.current=requestAnimationFrame(detect);
    return ()=>{ active=false; if(animRef.current) cancelAnimationFrame(animRef.current); };
  // eslint-disable-next-line
  },[cameraOn,mpStatus,detectionMode]);

  // ── Speech ─────────────────────────────────────────────────────────────────
  const speakSign = useCallback((text,force=false)=>{
    if(!text?.trim()) return;
    if(!force&&!settingsRef.current.voiceOutput) return;
    window.speechSynthesis.cancel();
    const translatedText = text.split(" ").map(w => translatePrediction(w, language)).join(" ");
    const u=new SpeechSynthesisUtterance(translatedText);
    const lmap={
      en: "en-US",
      hi: "hi-IN",
      te: "te-IN",
      ta: "ta-IN",
      kn: "kn-IN",
      ml: "ml-IN",
      es: "es-ES",
      fr: "fr-FR",
      de: "de-DE"
    };
    u.lang=lmap[language]||"en-US";
    u.rate=settingsRef.current.speechRate;
    u.pitch=settingsRef.current.pitch;
    u.volume=settingsRef.current.volume;
    u.onstart=()=>setSpeaking(true);
    u.onend=()=>setSpeaking(false);
    if(window.speechSynthesis.paused) window.speechSynthesis.resume();
    setTimeout(()=>window.speechSynthesis.speak(u),50);
  },[language]);

  useEffect(()=>{ speakRef.current=speakSign; },[speakSign]);

  const speakSentence=()=>speakSign(sentence.join(" "),true);
  const copyText=()=>{
    navigator.clipboard.writeText(sentence.join(" "));
    setCopied(true); setTimeout(()=>setCopied(false),2000);
  };

  // ── Text-to-Sign ───────────────────────────────────────────────────────────
  const t2sStart=useCallback(()=>{
    if(!t2sText.trim()) return;
    const chars=t2sText.toUpperCase().split("");
    if(t2sRef.current) clearInterval(t2sRef.current);
    setT2sAnim(true); setT2sIdx(0); let i=0;
    t2sRef.current=setInterval(()=>{
      i++;
      if(i>=chars.length){ clearInterval(t2sRef.current); t2sRef.current=null; setT2sAnim(false); setT2sIdx(-1); }
      else setT2sIdx(i);
    },1000/t2sSpeed);
  },[t2sText,t2sSpeed]);

  const t2sReset=useCallback(()=>{
    if(t2sRef.current) clearInterval(t2sRef.current);
    setT2sText(""); setT2sIdx(-1); setT2sAnim(false);
  },[]);

  // ── Active sign/conf for display ──────────────────────────────────────────
  const activeConf = wordConf||alphaConf;

  // ── Detection Page (inline for direct ref access) ─────────────────────────
  const DetectionPage = (
    <div className="detection-page">
      {/* Page header */}
      <div className="det-hdr">
        <div>
          <h2 className="page-title">{t("Live Sign Detection", language)}</h2>
          <p className="page-sub">{t("Real-time ASL recognition — ", language)}{mpStatus==="ready"?t("AI models active", language):t("Loading AI...", language)}</p>
        </div>

        {/* Manual Mode Selection Buttons */}
        <div className="mode-toggle-group">
          <button className={`mode-btn alphabet ${detectionMode==="alphabet"?"active":""}`}
            onClick={()=>{
              setDetectionMode("alphabet");
              seqBufRef.current=[]; lastVecRef.current=null;
              setSeqCount(0); setCurrentSign(""); setCurrentWord("");
              setAlphaConf(0); setWordConf(0); prevSignRef.current={sign:"",count:0};
              setFinalPrediction("—");
            }}>
            {t("Alphabet Mode", language)}
          </button>
          <button className={`mode-btn word ${detectionMode==="word"?"active":""}`}
            onClick={()=>{
              setDetectionMode("word");
              seqBufRef.current=[]; lastVecRef.current=null;
              setSeqCount(0); setCurrentSign(""); setCurrentWord("");
              setAlphaConf(0); setWordConf(0); prevSignRef.current={sign:"",count:0};
              setFinalPrediction("—");
            }}>
            {t("Word Mode", language)}
          </button>
        </div>

        <div className="det-hdr-right">
          {devices.length>0&&(
            <select className="cam-select" value={selDevice}
              onChange={e=>handleCamChange(e.target.value)}>
              {devices.map((d,i)=>(
                <option key={d.deviceId} value={d.deviceId}>{d.label||`${t("Camera", language)} ${i+1}`}</option>
              ))}
            </select>
          )}
          <button className={`icon-btn ${speaking?"active":""}`}
            onClick={()=>{const t_val=sentence.join(" ").trim();if(!t_val){alert(t("No sentence yet.", language));return;}speakSentence();}}>
            {speaking?<FiVolume2/>:<FiVolumeX/>}
          </button>
          <button className={`cam-btn ${cameraOn?"stop":"start"}`}
            onClick={cameraOn?stopCamera:startCamera}>
            {cameraOn?<><FiCameraOff/> {t("Stop Camera", language)}</>:<><FiCamera/> {t("Start Camera", language)}</>}
          </button>
        </div>
      </div>

      {/* Main body */}
      <div className="det-body">
        {/* ── Left: Camera Panel ── */}
        <div className="camera-panel">
          <div className="cam-panel-hdr">
            <span className="cam-panel-title"><FiActivity/> {t("LIVE HAND DETECTION", language)}</span>
            <div className="cam-panel-badges">
              {handCount>0&&<span className="hand-badge"><span className="hb-dot"/>✓ {t("Hand Detected", language)}</span>}
            </div>
          </div>

          <div className="cam-viewport">
            <video ref={videoRef} className="det-video" autoPlay muted playsInline/>
            <canvas ref={canvasRef} className="det-canvas"/>

            {/* Scanning corners */}
            <div className="sc tl"/><div className="sc tr"/><div className="sc bl"/><div className="sc br"/>

            {/* LIVE badge + timer */}
            {cameraOn&&(
              <div className="live-badge-new">
                <span className="live-red-dot"/>
                LIVE &nbsp; {fmtTime(sessTime)}
              </div>
            )}

            {/* Hand skeleton glow effect (when hand detected) */}
            {cameraOn&&handCount>0&&(
              <div className="hand-glow"/>
            )}

            {/* Detection Info Overlay */}
            {/* Recording overlay badge */}
            {cameraOn && isCapturing && (
              <div 
                className="recording-overlay-badge" 
                style={{
                  position: "absolute",
                  top: "20px",
                  left: "50%",
                  transform: "translateX(-50%)",
                  background: "rgba(16, 185, 129, 0.95)",
                  color: "#fff",
                  padding: "10px 20px",
                  borderRadius: "50px",
                  fontSize: "1.1rem",
                  fontWeight: "bold",
                  display: "flex",
                  alignItems: "center",
                  gap: "10px",
                  zIndex: 10,
                  boxShadow: "0 4px 15px rgba(16, 185, 129, 0.5)",
                  border: "1px solid rgba(255, 255, 255, 0.2)",
                  pointerEvents: "none",
                  letterSpacing: "0.5px"
                }}
              >
                <span 
                  style={{
                    display: "inline-block",
                    width: "10px",
                    height: "10px",
                    borderRadius: "50%",
                    background: "#fff",
                    animation: "pulse 1s infinite alternate"
                  }}
                />
                {t("CAPTURING SIGN GESTURE...", language)}
              </div>
            )}

            {/* Loading state */}
            {mpStatus==="loading"&&(
              <div className="mp-loading">
                <div className="mp-spinner"/>
                <p>{t("Loading AI Models...", language)}</p>
                <p className="mp-sub">MediaPipe HandLandmarker (WebGL)</p>
              </div>
            )}

            {/* Camera placeholder */}
            {!cameraOn&&(
              <div className="cam-empty">
                <MdSignLanguage className="cam-empty-icon"/>
                <p className="cam-empty-txt">
                  {mpStatus==="loading"?t("Loading AI models...", language):mpStatus==="error"?t("AI load failed — restart app", language):t("Click Start Camera to begin", language)}
                </p>
                {mpStatus==="ready"&&(
                  <button className="btn-primary btn-lg" onClick={startCamera}>
                    <FiCamera/> {t("Start Camera", language)}
                  </button>
                )}
              </div>
            )}
          </div>

          {/* Camera controls */}
          <div className="cam-controls">
            <button className="cc-btn" title="Screenshot" onClick={()=>{
              if(!canvasRef.current||!videoRef.current) return;
              const c=document.createElement("canvas");
              c.width=videoRef.current.videoWidth; c.height=videoRef.current.videoHeight;
              c.getContext("2d").drawImage(videoRef.current,0,0);
              const a=document.createElement("a"); a.download="sign.jpg";
              a.href=c.toDataURL("image/jpeg",0.9); a.click();
            }}><FiCamera/> {t("Screenshot", language)}</button>
            <button className="cc-btn" title="Reset sequence" onClick={()=>{
              seqBufRef.current=[]; lastVecRef.current=null;
              setSeqCount(0); setCurrentSign(""); setCurrentWord("");
              setAlphaConf(0); setWordConf(0); prevSignRef.current={sign:"",count:0};
            }}><FiLayers/> {t("Reset Sequence", language)}</button>
            <button className="cc-btn danger" onClick={stopCamera} title="Stop"><FiCameraOff/> {t("Stop", language)}</button>
          </div>

          {/* Live Camera Recording Test removed for production */}
        </div>

        {/* ── Right: Prediction Panel ── */}
        <div className="pred-panel">
          {/* Current prediction card */}
          <div className="glass-card pred-card">
            <div className="pred-card-hd">🎯 {t("CURRENT PREDICTION", language)}</div>
            <div className="pred-main" style={{display:"flex", alignItems:"center", justifyContent:"space-between", gap:"12px"}}>
              <div className="pred-split-layout" style={{display:"flex", gap:"16px", flex:1, minWidth:0}}>
                <div className="pred-sub-panel" style={{flex:1, minWidth:0}}>
                  <div className="pred-sign-lbl">{t("SIGN (Alphabet)", language)}</div>
                  <div className="pred-sign-val alpha" style={{fontSize:"2.4rem", fontWeight:"700", color:"var(--cyan)"}}>
                    {translatePrediction(currentSign, language) || "—"}
                  </div>
                  {cameraOn && !currentSign && (
                    <div className="pred-sub-status" style={{fontSize:"0.75rem", color:"var(--text-3)", marginTop:"4px"}}>
                      {t("No prediction", language)}
                      {detectionMode === "alphabet" && rejectionReason && (
                        <div style={{fontSize:"0.7rem", color:"var(--pink)", marginTop:"2px", fontStyle:"italic"}}>{rejectionReason}</div>
                      )}
                    </div>
                  )}
                </div>
                <div style={{width:"1px", background:"var(--border)", alignSelf:"stretch"}}/>
                <div className="pred-sub-panel" style={{flex:1, minWidth:0}}>
                  <div className="pred-sign-lbl">{t("WORD (WLASL)", language)}</div>
                  <div className="pred-sign-val word" style={{fontSize:currentWord.length>6?"1.5rem":"2rem", fontWeight:"700", color:"var(--purple)"}}>
                    {translatePrediction(currentWord, language) || "—"}
                  </div>
                  {cameraOn && !currentWord && (
                    <div className="pred-sub-status" style={{fontSize:"0.75rem", color:"var(--text-3)", marginTop:"4px"}}>
                      {t("No prediction", language)}
                      {detectionMode === "word" && rejectionReason && (
                        <div style={{fontSize:"0.7rem", color:"var(--pink)", marginTop:"2px", fontStyle:"italic"}}>{rejectionReason}</div>
                      )}
                    </div>
                  )}
                </div>
              </div>
              <CircularGauge value={activeConf} size={110}
                color={wordConf>0?"#a855f7":"#00d4ff"}/>
            </div>
          </div>

          {/* Automatic Sign Capture Word Mode Panel */}
          {detectionMode === "word" && (
            <div className="glass-card word-trigger-card" style={{ marginTop: "12px", padding: "16px", border: "1px solid var(--border)", borderRadius: "12px", background: "rgba(255, 255, 255, 0.02)" }}>
              <div className="card-hd" style={{ color: "var(--purple)", display: "flex", alignItems: "center", gap: "8px", fontSize: "0.95rem" }}>
                <FiZap /> {t("AUTOMATIC SIGN DETECTION", language)}
              </div>
              <div style={{ margin: "10px 0", fontSize: "0.85rem", color: "var(--text-2)", lineHeight: "1.4" }}>
                <p>
                  {t("Show your sign in the camera view. The system will automatically detect the hand movement and trigger a prediction when the sign is complete.", language)}
                </p>
                <div style={{ marginTop: "8px", padding: "8px", background: "rgba(168, 85, 247, 0.05)", borderLeft: "3px solid var(--purple)", borderRadius: "4px", fontSize: "0.8rem" }}>
                  <strong>{t("Status:", language)}</strong>{" "}
                  {cooldownActive ? (
                    <span style={{ color: "var(--pink)", fontWeight: "bold" }}>{t("COOLDOWN (Resetting...)", language)}</span>
                  ) : isCapturing ? (
                    <span style={{ color: "#10b981", fontWeight: "bold" }}>{t("CAPTURING SIGN GESTURE...", language)} ({seqCount}/{SEQ_LEN} frames)</span>
                  ) : (
                    <span>{t("Idle. Waiting for hand...", language)}</span>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Prediction history */}
          <div className="glass-card">
            <div className="pred-card-hd">📜 {t("PREDICTION HISTORY", language)}</div>
            <div className="pred-hist">
              {predHistory.length===0&&<p className="pred-empty">{t("No predictions yet...", language)}</p>}
              <AnimatePresence>
                {predHistory.map((p,i)=>(
                  <motion.span key={p.id}
                    className={`ph-chip ${p.isWord?"word":"alpha"}${i===0?" latest":""}`}
                    initial={{scale:0,opacity:0}} animate={{scale:1,opacity:1}}
                    exit={{scale:0,opacity:0}}>
                    {translatePrediction(p.sign, language)}
                  </motion.span>
                ))}
              </AnimatePresence>
            </div>
          </div>

          {/* Sentence builder */}
          <div className="glass-card">
            <div className="pred-card-hd"><FiZap/> {t("SENTENCE BUILDER", language)}</div>
            <div className="sentence-disp">
              {sentence.length===0
                ? <span className="sent-empty">{t("Start signing to build a sentence...", language)}</span>
                : <span className="sent-text">{formatSentence(sentence.map(w => translatePrediction(w, language)))}</span>
              }
            </div>
            <div className="sent-btns">
              <button className="sb-btn speak" onClick={speakSentence} disabled={!sentence.length}>
                <FiVolume2/> {speaking?t("Speaking...", language):t("Speak", language)}
              </button>
              <button className="sb-btn copy" onClick={copyText} disabled={!sentence.length}>
                {copied?<FiCheck/>:<FiCopy/>} {copied?t("Copied!", language):t("Copy", language)}
              </button>
              <button className="sb-btn space-btn" onClick={() => setSentence(p => [...p, " "])} disabled={!cameraOn}>
                + {t("Space", language)}
              </button>
              <button className="sb-btn clear" onClick={()=>{setSentence([]);lastAddRef.current={sign:"",time:0};}} disabled={!sentence.length}>
                <FiTrash2/> {t("Clear", language)}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* ── Stats Bar ── */}
      <div className="stats-bar">
        <StatMini icon={<MdSpeed/>} value={fps} label={t("FPS", language)}
          sub={fps>=20?t("Excellent", language):fps>=10?t("Good", language):t("Low", language)}
          color={fpsColor(fps)}/>
        <StatMini icon={<MdSignLanguage/>} value={handCount} label={t("Hand(s)", language)}
          sub={handCount>0?t("Detected", language):t("None", language)}
          color={handCount>0?"var(--green)":"var(--text-3)"}/>
        <StatMini icon={<FiTarget/>} value={`${activeConf.toFixed(1)}%`} label={t("Confidence", language)}
          sub={t(confLabel(activeConf), language)} color="var(--cyan)"/>
        <StatMini icon={<FiLayers/>} value={`${seqCount}/${SEQ_LEN}`} label={t("Sequence", language)}
          sub={seqCount===SEQ_LEN?t("Complete", language):t("Building", language)}
          color={seqCount===SEQ_LEN?"var(--green)":"var(--purple)"}/>
        <StatMini icon={<FiShield/>} value={`${stability}%`} label={t("Stability", language)}
          sub={t(qualityLabel(stability), language)}
          color={stability>=80?"var(--green)":"var(--orange)"}/>
        <StatMini icon={<FiZap/>} value={latency} unit="ms" label={t("Latency", language)}
          sub={t(latLabel(latency), language)}
          color={latency<50?"var(--green)":latency<120?"var(--orange)":"var(--pink)"}/>
      </div>

      {/* ── System status footer ── */}
      <div className="sys-footer">
        <div className="sf-status">
          <span className={`sf-dot ${apiStatus==="online"&&mpStatus==="ready"?"green":"orange"}`}/>
          {t("System Status", language)}:&nbsp;
          <strong>
            {apiStatus==="online"&&mpStatus==="ready"
              ? t("All Systems Operational", language)
              : mpStatus==="loading"
              ? t("Loading AI models...", language)
              : t("Partial — check backend", language)}
          </strong>
        </div>
        <span className="sf-time">{new Date().toLocaleString([],{dateStyle:"medium",timeStyle:"short"})}</span>
      </div>
    </div>
  );

  // ── Render ─────────────────────────────────────────────────────────────────
  const renderPage = ()=>{
    switch(page){
      case "home":      return <HomePage setPage={setPage} analytics={analytics} language={language}/>;
      case "detection": return DetectionPage;
      case "text2sign": return <Text2SignPage inputText={t2sText} setInputText={setT2sText}
          idx={t2sIdx} isAnim={t2sAnim} speed={t2sSpeed} setSpeed={setT2sSpeed}
          start={t2sStart} reset={t2sReset} speak={speakSign} language={language}/>;
      case "analytics": return <AnalyticsPage analytics={analytics} fmtSessTime={fmtTime} language={language}/>;
      case "emergency": return <EmergencyPage settings={settings} setSettings={setSettings} speak={speakSign} language={language}/>;
      case "about":     return <AboutPage language={language}/>;
      case "settings":  return <SettingsPage language={language} setLanguage={setLanguage}
          settings={settings} setSettings={setSettings}/>;
      default:          return <HomePage setPage={setPage} analytics={analytics} language={language}/>;
    }
  };

  return (
    <div className={`shell${settings.highContrast?" hi-con":""}`}>
      {/* ── Sidebar ── */}
      <aside className={`sidebar ${sidebar?"open":"closed"}`}>
        <div className="sb-brand">
          <div className="sb-logo"><MdSignLanguage/></div>
          {sidebar&&(
            <div className="sb-brand-txt">
              <p className="sb-name">SilentSpeak</p>
              <p className="sb-sub">AI Sign Language System</p>
            </div>
          )}
        </div>

        <nav className="sb-nav">
          {NAV.map(item=>(
            <button key={item.id} className={`nav-item${page===item.id?" active":""}`}
              onClick={()=>setPage(item.id)}>
              <item.icon className="nav-icon"/>
              {sidebar&&<span className="nav-label">{item.label}</span>}
              {cameraOn&&item.id==="detection"&&sidebar&&<span className="nav-live-dot"/>}
            </button>
          ))}
        </nav>

        {sidebar&&(
          <div className="sb-footer">
            <div className={`sb-api-badge ${apiStatus}`}>
              {apiStatus==="online"?<FiWifi/>:<FiWifiOff/>}
              {apiStatus==="online"?"Backend Live":"Backend Offline"}
            </div>

            <div className={`sb-mp-badge ${mpStatus}`}>
              {mpStatus==="ready"?"🟢":"🔴"} MediaPipe {mpStatus==="ready"?"Ready":mpStatus==="loading"?"Loading...":"Error"}
            </div>
          </div>
        )}

        <button className="sb-toggle" onClick={()=>setSidebar(s=>!s)}>
          {sidebar?<FiChevronLeft/>:<FiChevronRight/>}
        </button>
      </aside>

      {/* ── Main area ── */}
      <div className="main-area">
        {/* Topbar */}
        <header className="topbar">
          <div className="tb-left">
            <div className="tb-search">
              <FiSearch className="tb-search-icon"/>
              <span>{t("Real-time AI Sign Detection", language)}</span>
              {cameraOn&&<span className="tb-active-dot"/>}
            </div>
          </div>
          <div className="tb-right">
            <div className="fps-pill" style={{color:fpsColor(fps)}}><FiActivity/> {t("FPS", language)}: {fps}</div>
            <div className={`status-pill ${apiStatus}`}>
              {apiStatus==="online"?<FiWifi/>:<FiWifiOff/>}
              {apiStatus==="online"?t("Active", language):t("Offline", language)}
              {apiStatus==="online"&&<span className="status-pulse"/>}
            </div>
            <div className="lang-pill" onClick={()=>setPage("settings")}>
              <FiGlobe/> {LANGUAGES.find(l=>l.code===language)?.label}
            </div>
          </div>
        </header>

        {/* Content */}
        <main className="content">
          <AnimatePresence mode="wait">
            <motion.div key={page}
              initial={{opacity:0,y:14}} animate={{opacity:1,y:0}}
              exit={{opacity:0,y:-14}} transition={{duration:0.18}}
              style={{height:"100%"}}>
              {renderPage()}
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
    </div>
  );
}