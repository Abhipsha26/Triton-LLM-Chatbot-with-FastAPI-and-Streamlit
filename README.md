# Triton-LLM-Chatbot-with-FastAPI-and-Streamlit
Interactive chatbot demo using NVIDIA Triton Inference Server, FastAPI, and Streamlit. User queries from a web UI are sent to a deployed LLM via FastAPI, enabling real-time conversational AI in a modular, research-friendly setup.

Project Overview
This project demonstrates how to deploy and interact with a large language model (LLM) using NVIDIA Triton Inference Server, FastAPI, and Streamlit. The goal is to provide a modular, research-friendly chatbot pipeline that showcases how modern LLMs can be served and accessed via a web interface.

Components and Workflow
1. Model Deployment with Triton Inference Server

The LLM (e.g., Phi-1.5) is deployed using NVIDIA Triton Inference Server in a Docker container.

The model is loaded from a local cache and made available for inference via Triton’s HTTP API.

2. FastAPI Backend

A FastAPI application acts as an intermediary between the web UI and Triton.

It receives user queries, formats them as Triton inference requests, sends them to the server, and returns the model’s responses.

3. Streamlit Chatbot UI

A Streamlit app provides a simple, interactive web-based chatbot interface.

Users can type messages, which are sent to the FastAPI backend and then to the LLM.

Responses from the model are displayed in the chat interface.

How it Works
User Interaction:
The user enters a message in the Streamlit chatbot interface.

API Request:
Streamlit sends the user’s message to the FastAPI backend via a POST request.

Model Inference:
FastAPI forwards the request to the Triton Inference Server, which runs the LLM and generates a response.

Response Delivery:
The model’s response is sent back through FastAPI to the Streamlit UI and displayed to the user.

Why This Project?
Modular: Each component (Triton, FastAPI, Streamlit) can be developed and improved independently.

Demonstrative: Shows a full pipeline from model serving to user interaction.

Research-Friendly: Designed for experimentation, not production use.

Extensible: Can be adapted for other models, UIs, or API frameworks.
