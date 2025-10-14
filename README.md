### Code Structure
```
---
repo/
├── configs/        # YAML configs for datasets and model parameters
├── data/           # Dataset processing and normalization scripts
├── models/         # Model definitions (WAP, CoMER, Qwen2.5-VL)
├── utils/          # Logging, metrics, visualization helpers
├── lit_model.py    # Main model
├── train.py        # Main training entry point
├── eval.py         # Evaluation on CROHME and HME100K
├── demo_app.py     # Gradio UI demo interface
├── notebooks/      # Interactive Jupyter demos
└── README.md       # Documentation and setup guide
└── requirements.txt 
---
```