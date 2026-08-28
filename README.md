# 👁️ YOLO Based Eye Tracking System for Real-Time CVI

## A Webcam-Based Gaze-Controlled Ball Game Using YOLO11n

---

## 👥 Team C10

| Team Member | Roll Number |
|---|---|
| **Gandrothu Karthik** | **CB.SC.U4AIE24217** |
| **Karlapati Sreeshanth** | **CB.SC.U4AIE24225** |
| **Namburi Rupesh** | **CB.SC.U4AIE24234** |
| **Uday Sri Yaramati** | **CB.SC.U4AIE24260** |

**Course:** Deep Learning  
**Course Code:** 22AIE304  
**Institution:** Amrita Vishwa Vidyapeetham

---

## 📌 About the Project

This project is a real-time eye tracking and gaze-controlled game system that uses a standard webcam and a trained **YOLO11n model** to detect pupils.

The main idea is simple: instead of using expensive and specialized eye-tracking hardware, the system uses a normal webcam to capture the user's eyes, detect the pupils, calculate their position, estimate the direction of gaze, and use that direction to control a ball inside a Pygame window.

The user can control the ball using only their eye movement.

| Gaze Direction | Game Action |
|---|---|
| 👈 Look Left | Ball moves Left |
| 👉 Look Right | Ball moves Right |
| 👆 Look Up | Ball moves Up |
| 👇 Look Down | Ball moves Down |
| 👁️ Look Center | Center / Neutral |
| ❌ No Pupil Detected | No gaze-based movement |

The project is motivated by the need for a low-cost and accessible gaze-based interaction system, particularly for visual activities related to **Cerebral Visual Impairment (CVI)**.

---

## 🧠 Project Motivation

Cerebral Visual Impairment (CVI) is a condition where the brain has difficulty processing visual information. People with CVI may experience difficulties with focusing, tracking, and interpreting visual objects.

Traditional eye-tracking systems can require specialized hardware and can be expensive.

This motivated us to explore a simpler approach:

> **Can we use a normal webcam and deep learning to convert eye movement into an interactive game control?**

Our project combines:

```text
Standard Webcam
       ↓
YOLO11n Pupil Detection
       ↓
Pupil Position
       ↓
Gaze Direction Estimation
       ↓
Dead Zone + Smoothing
       ↓
Directional Command
       ↓
Pygame Ball Control
