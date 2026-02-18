import requests
import streamlit as st
import time
import os
from dotenv import load_dotenv

# Load .env (for local development)
load_dotenv()

# =============================================================================
# TOKEN MANAGEMENT (SECURE)
# =============================================================================

def get_hf_token():
    """Securely retrieve Hugging Face token"""

    # 1. Streamlit Cloud
    try:
        return st.secrets["HF_TOKEN"]
    except (KeyError, FileNotFoundError):
        pass

    # 2. Environment variable (.env or system env)
    token = os.getenv("HF_TOKEN")
    if token:
        return token

    return None


# =============================================================================
# CLOUD AI - HUGGING FACE
# =============================================================================

def chat_with_huggingface(message, context=""):
    """
    Chat using Hugging Face API (cloud-based)
    Returns structured response dict
    """

    HF_TOKEN = get_hf_token()

    if not HF_TOKEN:
        return {
            "success": False,
            "response": None,
            "error": "Hugging Face token not configured."
        }

    API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    system_prompt = """
You are a media literacy assistant.
Your role:
- Explain misinformation clearly
- Encourage verification
- Avoid certainty claims
- Keep responses concise (max 3 paragraphs)

Never claim information is 100% true or false.
Always encourage cross-checking with reliable sources.
"""

    if context:
        system_prompt += f"\n\nContext from analysis:\n{context}"

    full_prompt = f"{system_prompt}\n\nUser: {message}\n\nAssistant:"

    payload = {
        "inputs": full_prompt,
        "parameters": {
            "max_new_tokens": 300,
            "temperature": 0.7,
            "top_p": 0.9,
            "return_full_text": False
        }
    }

    # Retry logic (3 attempts)
    for attempt in range(3):
        try:
            response = requests.post(
                API_URL,
                headers=headers,
                json=payload,
                timeout=30
            )

            # Model loading
            if response.status_code == 503:
                time.sleep(5)
                continue

            response.raise_for_status()
            result = response.json()

            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get("generated_text", "").strip()
            elif isinstance(result, dict):
                generated_text = result.get("generated_text", "").strip()
            else:
                return {
                    "success": False,
                    "response": None,
                    "error": "Unexpected API response format."
                }

            return {
                "success": True,
                "response": generated_text,
                "error": None
            }

        except requests.exceptions.Timeout:
            if attempt < 2:
                time.sleep(3)
                continue
            return {
                "success": False,
                "response": None,
                "error": "Request timed out."
            }

        except requests.exceptions.HTTPError as e:
            return {
                "success": False,
                "response": None,
                "error": f"HTTP Error {e.response.status_code}"
            }

        except Exception as e:
            return {
                "success": False,
                "response": None,
                "error": str(e)
            }

    return {
        "success": False,
        "response": None,
        "error": "Model loading timeout."
    }


# =============================================================================
# LOCAL AI - OLLAMA
# =============================================================================

def is_ollama_available():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def chat_with_ollama(message, context=""):
    """
    Chat using local Ollama
    Returns structured response dict
    """

    if not is_ollama_available():
        return {
            "success": False,
            "response": None,
            "error": "Ollama not running."
        }

    url = "http://localhost:11434/api/generate"

    system_prompt = """
You are a media literacy assistant.
Keep responses concise and educational.
"""

    if context:
        system_prompt += f"\n\nContext:\n{context}"

    full_prompt = f"{system_prompt}\n\nUser: {message}\n\nAssistant:"

    payload = {
        "model": "llama3.2:3b",
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 300
        }
    }

    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()

        result = response.json()

        return {
            "success": True,
            "response": result.get("response", ""),
            "error": None
        }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "response": None,
            "error": "Ollama timeout."
        }

    except Exception as e:
        return {
            "success": False,
            "response": None,
            "error": str(e)
        }


# =============================================================================
# HYBRID SYSTEM WITH CLEAN FALLBACK
# =============================================================================

def get_ai_response(message, context="", prefer_local=False):
    """
    Hybrid AI system with structured fallback
    Returns (text_response, source_label)
    """

    if prefer_local:

        # Try local first
        local_response = chat_with_ollama(message, context)

        if local_response["success"]:
            return local_response["response"], "local"

        # Fallback to cloud
        cloud_response = chat_with_huggingface(message, context)

        if cloud_response["success"]:
            return cloud_response["response"], "cloud (fallback)"

        return cloud_response["error"], "cloud (fallback failed)"

    else:

        # Try cloud first
        cloud_response = chat_with_huggingface(message, context)

        if cloud_response["success"]:
            return cloud_response["response"], "cloud"

        # Fallback to local
        local_response = chat_with_ollama(message, context)

        if local_response["success"]:
            return local_response["response"], "local (fallback)"

        return cloud_response["error"], "cloud (failed)"


# =============================================================================
# AI EXPLANATION GENERATION
# =============================================================================

def generate_ai_explanation(text, prediction, credibility, flags, prefer_local=False):
    """
    Generate AI explanation of classification results
    """

    verdict = "likely real news" if prediction == 1 else "likely fake news"

    context = f"""
Analyzed text (excerpt):
"{text[:200]}..."

Classification: {verdict}
Credibility score: {credibility:.1f}%
Red flags detected: {', '.join(flags) if flags else 'none'}
"""

    question = f"""
Based on the analysis results, why was this content classified as {verdict}?
Explain in simple terms what patterns were detected.
Keep it brief (2-3 sentences).
"""

    return get_ai_response(question, context, prefer_local)


# =============================================================================
# OPTIONAL: SIMPLE RATE LIMITING (CALL IN STREAMLIT APP)
# =============================================================================

def check_rate_limit(seconds=3):
    if "last_call" not in st.session_state:
        st.session_state.last_call = 0

    if time.time() - st.session_state.last_call < seconds:
        return False

    st.session_state.last_call = time.time()
    return True


# =============================================================================
# TEST FUNCTION
# =============================================================================

def test_chatbot():
    print("Testing Chatbot Module")
    print("=" * 50)

    response, source = get_ai_response("What is fake news?")
    print(f"Source: {source}")
    safe_response = (response or "")[:200]
    print(f"Response: {safe_response}...")

    print("=" * 50)
    print("Test complete.")


if __name__ == "__main__":
    test_chatbot()
