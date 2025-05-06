# TBPK-Project - Video Transcript RAG - Ollama & Qdrant
This project is a Retrieval Augmented Generation (RAG) pipeline which enables users to ask our local language model, using a vector similarity search via Qdrant, various questions relating to the CS370 course material. Our model will then return related video segments that will answer the question the user posed.

Created by **Tomasz Brauntsch & Preston Khorosh**

---

# Features & Capabilities
- Answering questions via video segments
- Subtitle chunking and segmenting using spaCy
- BERTopic topic modeling
- Frontend via Gradio

---

# Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/PretzonPretzel/TBPK-Project.git
cd TBPK-Project
```

### 2. Confirm qDrant
Confirm that qDrant is running on port **6333** and has the correct URL
```python
client = QdrantClient(url="http://eng-ai-agents-qdrant-1:6333")
```

### 3. Download requirements
```bash
pip install -r requirements.txt
```

### 4. Run & Access
```bash
python3 main.py
```
Access with **http://localhost:7860**

---

# Demo Screenshots
![image](https://github.com/user-attachments/assets/1911d8ab-7667-400d-b5a0-d67edfafea62)

![image](https://github.com/user-attachments/assets/48f6dc08-761f-49e0-bf75-ad05b5ad1c39)

![image](https://github.com/user-attachments/assets/df1a7b73-13ce-4232-a479-0967a9d501b3)
