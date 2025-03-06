A simple chatbot trained on Red Dwarf

Steps to Train a Talkie Toaster on Red Dwarf Scripts
1. Prepare Your Dataset
    - Collect and clean your TV script dialogues.
    - Save it as a .txt or .json file with clearly defined conversations.

2. Tokenize & Preprocess the Data
    - Load the script text.
    - Tokenize using AutoTokenizer.
    - Convert it into training-friendly format (e.g., padding, truncation).

3. Fine-Tune distilgpt2 on Your Data
    - Use Trainer API or manually train with torch.optim.
    - Save the trained model.

4. Test & Evaluate the Model
    - Generate responses based on sample inputs.
        - COMPLETE: Initial test that model loads and responds but without training
    - Check if it mimics the show's dialogue style.

5. Deploy as a Chatbot
    - Load the fine-tuned model.
    - Create a chatbot pipeline with pipeline("text-generation").
    - Serve it via a web app (Flask, FastAPI, or React frontend).