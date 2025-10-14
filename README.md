# DSP391m_Group1_FALL25

# 🧮 Handwritten Mathematical Expression Recognition (HMER)  
### Data Science Project — FPT University | DSP391-FA25-AI1802


### <b><span style="color:red">NOTE: LOOK AT THE DIFFERENCE BRANCH TO SEE THE DIFFERENCE CODE MODEL</span></b>

---

## 🇬🇧 English Version

### 📘 Overview
This project aims to develop an **end-to-end system for recognizing handwritten mathematical expressions (HMER)** and converting them into valid **LaTeX** strings.  
It was conducted as part of the **Data Science Project (DSP391m)** course at **FPT University**, HCMC campus.

Handwritten mathematical recognition is much more complex than standard OCR, due to:
- **2D spatial structures** (fractions, subscripts/superscripts, roots)
- **High handwriting variability**
- **Symbol ambiguity**
- **Strict LaTeX grammar requirements**

We explore and compare **three model families**:
1. **WAP (CNN + RNN Attention)** – baseline sequence-to-sequence model  
2. **CoMER (Transformer + Coverage)** – modern Transformer with attention refinement  
3. **Qwen2.5-VL (Vision-Language Model fine-tuning)** – SFT approach for multimodal adaptation  

---

### 🎯 Objectives
- Convert handwritten expression images into structurally correct LaTeX formulas.  
- Achieve strong benchmark scores on **CROHME** and **HME100K** datasets.  
- Provide model comparisons with detailed quantitative and qualitative analysis.  
- Deliver a reproducible public codebase with configuration and logs via **Weights & Biases (W&B)**.  
- Deploy a runnable **Gradio demo** for public interaction.  

---

### 🧩 Methodology
**Datasets Used:**  
- **HME100K** — large-scale offline handwritten dataset for pretraining.  
- **CROHME** — standard academic benchmark for fine-tuning and evaluation.  

**Training Strategy:**  
- Stage A: Pretraining on HME100K (warm-up / full).  
- Stage B: Fine-tuning on CROHME (multiple years).  
- Stage C: Evaluation on benchmark datasets, with robustness tests (noise, blur, etc.).  

**Metrics:**  
- Expression Recognition Rate (**ExpRate**)  
- Character / Word Error Rate (**CER**, **WER**)  
- Token accuracy, latency (ms)  

**Experiment Logging:**  
All training sessions and results are tracked with **Weights & Biases**, using YAML/JSON configs for full reproducibility.

---

### 🏗️ System Design
**Pipeline:**
1. Data Loading → Normalization → Augmentation  
2. Model Training (WAP / CoMER / Qwen2.5-VL)  
3. LaTeX Validator (rule-based syntax check)  
4. Evaluation on CROHME / HME100K  
5. Demo Deployment via Gradio  


### Code Structure
```
---
repo/
├── configs/        # YAML configs for datasets and model parameters
├── data/           # Dataset processing and normalization scripts
├── models/         # Model definitions (WAP, CoMER, Qwen2.5-VL)
├── utils/          # Logging, metrics, visualization helpers
├── train.py        # Main training entry point
├── eval.py         # Evaluation on CROHME and HME100K
├── demo_app.py     # Gradio UI demo interface
├── notebooks/      # Interactive Jupyter demos
└── README.md       # Documentation and setup guide
---
```


### 🧪 Results & Discussion
- Comparative study: WAP vs. CoMER vs. Qwen2.5  
- Ablation: warm-up vs. full pretraining  
- Robustness under visual perturbations (noise, blur)  
- Demo performance (inference latency, UX feedback)  

**Future Work:**
- Grammar-aware beam search decoding  
- Curriculum learning with multi-dataset scheduling  
- Scaling to larger multimodal vision-language models  



### 👥 Team Members
| Name | Role |
|------|------|
| **Nguyen Minh Khoa** | Leader – Modeling & Training Lead |
| **Phan Van Hai Nam** | Data & EDA, Preprocessing, W&B Pipeline |
| **Nguyen Ngoc Thien Phu** | Demo/UI, Evaluation, Documentation |

**Supervisor:** Capstone Project DSP391m — FPT University  



### ⚖️ Ethics & Licensing
All datasets used are **publicly available** (CROHME, HME100K).  
No personal data was collected.  
Outputs are limited to mathematical LaTeX strings.  
This project is for **academic research only**, in compliance with DSP391m ethics standards.



### 📚 References
1. Zhang et al., *Watch, Attend and Parse: An End-to-End Neural Network Based Approach to Handwritten Mathematical Expression Recognition*, Pattern Recognition, 2017.  
2. Zhao et al., *CoMER: Modeling Coverage for Transformer-based Handwritten Mathematical Expression Recognition*, ECCV 2022.  
3. Mouchère et al., *Competition on Recognition of Online Handwritten Mathematical Expressions (CROHME)*, ICDAR 2014.  
4. Phymond et al., *HME100K: A Large-Scale Real-Scene Offline Dataset for Handwritten Mathematical Expressions*, GitHub, 2021.  



## 🇻🇳 Phiên bản tiếng Việt

### <b><span style="color:red">GHI CHÚ: MỖI 1 MODEL ĐƯỢC ĐỂ TRONG TỪNG BRANCH KHÁC NHAU, XIN HÃY CHUYỂN BRANCH ĐỂ XEM TỪNG REPO RÕ HƠN</span></b>


### 📘 Tổng quan
Dự án này xây dựng **hệ thống nhận dạng biểu thức toán học viết tay (HMER)** từ hình ảnh và chuyển đổi sang **mã LaTeX hợp lệ**.  
Được thực hiện trong khuôn khổ môn học **Đồ án Khoa học Dữ liệu (DSP391m)** tại **Đại học FPT**, cơ sở TP.HCM.

Bài toán HMER phức tạp hơn OCR văn bản thông thường vì:
- Cấu trúc **hai chiều** (phân số, chỉ số, căn, v.v.)  
- **Nhiều kiểu chữ viết tay khác nhau**  
- **Dễ nhầm ký hiệu**  
- **Quy tắc cú pháp LaTeX nghiêm ngặt**  

Nhóm tiến hành thử nghiệm với **ba họ mô hình chính**:
1. **WAP (CNN + RNN Attention)** – mô hình cơ sở  
2. **CoMER (Transformer + Coverage)** – mô hình hiện đại dựa trên attention refinement  
3. **Qwen2.5-VL (Vision-Language Fine-tuning)** – tinh chỉnh mô hình thị giác-ngôn ngữ hiện đại  

---

### 🎯 Mục tiêu
- Chuyển đổi ảnh biểu thức toán viết tay thành chuỗi LaTeX hợp lệ.  
- Đạt độ chính xác cao trên bộ dữ liệu **CROHME** và **HME100K**.  
- Phân tích, so sánh và đánh giá ưu/nhược điểm của từng mô hình.  
- Cung cấp **mã nguồn công khai**, có khả năng tái hiện kết quả.  
- Triển khai **demo trực quan với Gradio** để trình diễn hệ thống.  

---

### 🧩 Phương pháp
**Dữ liệu huấn luyện:**  
- **HME100K** (ngoại tuyến, quy mô lớn) – dùng cho giai đoạn tiền huấn luyện.  
- **CROHME** – dùng cho tinh chỉnh và đánh giá mô hình.  

**Chiến lược huấn luyện:**  
- Giai đoạn A: Pretrain trên HME100K (warm-up / full).  
- Giai đoạn B: Fine-tune trên CROHME.  
- Giai đoạn C: Đánh giá và kiểm tra độ bền mô hình.  

**Chỉ số đánh giá:**  
- **ExpRate (Expression Recognition Rate)**  
- **CER/WER**, độ chính xác token, tốc độ xử lý.  

**Theo dõi thí nghiệm:**  
Sử dụng **Weights & Biases (W&B)** để log kết quả và checkpoint, cấu hình YAML/JSON đảm bảo tái hiện được toàn bộ pipeline.

---

### 🏗️ Thiết kế hệ thống
**Quy trình tổng quát:**
1. Tiền xử lý & nạp dữ liệu  
2. Huấn luyện mô hình (WAP / CoMER / Qwen2.5-VL)  
3. Kiểm tra cú pháp LaTeX  
4. Đánh giá kết quả trên CROHME và HME100K  
5. Triển khai demo qua Gradio  

**Cấu trúc mã nguồn:**
```
repo/
├── configs/        # Cấu hình tham số
├── data/           # Tiền xử lý dữ liệu
├── models/         # Mô hình huấn luyện
├── utils/          # Tiện ích logging, metric
├── train.py        # Huấn luyện chính
├── eval.py         # Đánh giá
├── demo_app.py     # Giao diện demo
└── notebooks/      # Notebook minh họa
└──  README.md      # Hướng dẫn đọc và setup
```

---

### 🧪 Kết quả & Thảo luận
- So sánh hiệu năng giữa WAP – CoMER – Qwen2.5  
- Đánh giá ảnh hưởng của chiến lược pretraining  
- Thử nghiệm độ bền (nhiễu Gaussian, mờ, giảm chất lượng ảnh)  
- Đánh giá thời gian phản hồi demo  

**Hướng mở rộng:**
- Khảo sát mô hình LLM đa phương thức quy mô lớn hơn  

---

### 👥 Thành viên
| Họ và tên | Vai trò |
|------------|----------|
| **Nguyễn Minh Khoa** | Trưởng nhóm – Huấn luyện và tích hợp mô hình |
| **Phan Văn Hải Nam** | Xử lý dữ liệu, EDA, pipeline theo dõi W&B |
| **Nguyễn Ngọc Thiên Phú** | Phát triển giao diện Gradio, đánh giá và tài liệu hóa |

**Giảng viên hướng dẫn:** Môn DSP391m – Đại học FPT  

---

### ⚖️ Đạo đức & Bản quyền
- Tất cả dữ liệu (CROHME, HME100K) đều **công khai và hợp pháp**.  
- Không thu thập dữ liệu cá nhân.  
- Sản phẩm chỉ phục vụ **mục đích nghiên cứu học thuật**.  

---

### 📚 Tài liệu tham khảo
1. Zhang et al., *Watch, Attend and Parse*, Pattern Recognition, 2017.  
2. Zhao et al., *CoMER: Modeling Coverage for Transformer-based HMER*, ECCV 2022.  
3. Mouchère et al., *CROHME Dataset*, ICDAR 2014.  
4. Phymond et al., *HME100K Dataset*, GitHub 2021.  

---

> 📌 *This project is part of FPT University’s Data Science Capstone Program — designed for exploring deep learning solutions to complex OCR challenges in academic and industrial research.*

