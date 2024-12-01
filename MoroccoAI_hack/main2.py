from flask import Flask, render_template, request, jsonify
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoProcessor,
    AutoModelForImageTextToText
)
import torch
import os
from datetime import datetime
import logging
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app configuration
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load translation models
logger.info("Loading translation models...")
darija_to_english_tokenizer = AutoTokenizer.from_pretrained("ychafiqui/darija-to-english-2")
darija_to_english_model = AutoModelForSeq2SeqLM.from_pretrained("ychafiqui/darija-to-english-2")
english_to_darija_tokenizer = AutoTokenizer.from_pretrained("atlasia/Terjman-Ultra")
english_to_darija_model = AutoModelForSeq2SeqLM.from_pretrained("atlasia/Terjman-Ultra")

# Load vision-language models
logger.info("Loading vision-language models...")
processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision")
vision_model = AutoModelForImageTextToText.from_pretrained("meta-llama/Llama-3.2-11B-Vision")

# Setup devices
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
darija_to_english_model = darija_to_english_model.to(device)
english_to_darija_model = english_to_darija_model.to(device)
vision_model = vision_model.to(device)


# Translation utilities
def translate_darija_to_english(darija_text):
    try:
        input_tokens = darija_to_english_tokenizer(darija_text, return_tensors="pt", padding=True, truncation=True).to(device)
        output_tokens = darija_to_english_model.generate(**input_tokens)
        return darija_to_english_tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    except Exception as e:
        logger.error(f"Error in Darija to English translation: {e}")
        return darija_text


def translate_english_to_darija(english_text):
    try:
        input_tokens = english_to_darija_tokenizer(english_text, return_tensors="pt", padding=True, truncation=True).to(device)
        output_tokens = english_to_darija_model.generate(**input_tokens)
        return english_to_darija_tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    except Exception as e:
        logger.error(f"Error in English to Darija translation: {e}")
        return english_text


# AI Agents
class PregnancyAgent:
    def __init__(self):
        self.processor = processor
        self.model = vision_model

    def _run_model(self, prompt: str, image: Image.Image = None) -> str:
        try:
            if image:
                inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(device)
            else:
                inputs = self.processor(text=prompt, return_tensors="pt").to(device)

            outputs = self.model.generate(**inputs)
            return self.processor.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Error in running model: {e}")
            return "عذراً، حدث خطأ أثناء معالجة طلبك."

    def process_image(self, image_path: str) -> Image.Image:
        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            logger.error(f"Error in processing image: {e}")
            return None


# Specialized agents
class DietaryGuidanceAgent(PregnancyAgent):
    def respond(self, history, darija_prompt, image=None):
        english_prompt = translate_darija_to_english(darija_prompt)
        system_prompt = f"You are a nutritional specialist. {english_prompt}"

        english_response = self._run_model(system_prompt, image)
        darija_response = translate_english_to_darija(english_response)

        history.append([darija_prompt, darija_response])
        return history


class QuestionAnsweringAgent(PregnancyAgent):
    def respond(self, history, darija_prompt, image=None):
        english_prompt = translate_darija_to_english(darija_prompt)
        system_prompt = f"You are a pregnancy health specialist. {english_prompt}"

        english_response = self._run_model(system_prompt, image)
        darija_response = translate_english_to_darija(english_response)

        history.append([darija_prompt, darija_response])
        return history


class PostPregnancyAgent(PregnancyAgent):
    def respond(self, history, darija_prompt, image=None):
        english_prompt = translate_darija_to_english(darija_prompt)
        system_prompt = f"You are a postpartum care specialist. {english_prompt}"

        english_response = self._run_model(system_prompt, image)
        darija_response = translate_english_to_darija(english_response)

        history.append([darija_prompt, darija_response])
        return history


# Chatbot backend
class ChatbotBackend:
    def __init__(self):
        self.agents = {
            'الإرشاد الغذائي': DietaryGuidanceAgent(),
            'الأسئلة العامة': QuestionAnsweringAgent(),
            'دعم ما بعد الولادة': PostPregnancyAgent()
        }
        self.chat_history = []

    def get_response(self, darija_prompt, agent_type, image_path=None):
        agent = self.agents.get(agent_type)
        if not agent:
            return "عذراً، لم يتم العثور على الوكيل المطلوب."

        image = agent.process_image(image_path) if image_path else None
        return agent.respond(self.chat_history, darija_prompt, image)[-1][1]


chatbot = ChatbotBackend()

# Flask routes
@app.route('/')
def home():
    return render_template('welcome.html')


@app.route('/send_message', methods=['POST'])
def send_message():
    message = request.form.get('message', '')
    agent_type = request.form.get('agent_type', 'الإرشاد الغذائي')
    image = request.files.get('image')
    image_path = None

    if image:
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{image.filename}"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(image_path)

    response = chatbot.get_response(message, agent_type, image_path)

    if image_path and os.path.exists(image_path):
        os.remove(image_path)

    return jsonify({'response': response})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
