from flask import Flask, render_template, request
from dotenv import load_dotenv
import os
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes

load_dotenv()

api_key = os.getenv("WATSONX_API_KEY")
project_id = os.getenv("WATSONX_PROJECT_ID")
url = os.getenv("WATSONX_URL")

credentials = {
    "apikey": api_key,
    "url": url
}

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    story = ""
    if request.method == "POST":
        prompt = request.form.get("prompt", "")
        genre = request.form.get("genre", "")

        full_prompt = f"""
You are a creative AI writer. Write a short story based on the following prompt:

"{prompt}"

Make it 3–5 paragraphs long. Include vivid characters, emotions, and creative events. 
Avoid repetition. {f"The story should be in the {genre} genre." if genre else ""}
"""

        try:
            model = Model(
                model_id="meta-llama/llama-3-3-70b-instruct",
                params={
    GenParams.DECODING_METHOD: "sample",
    GenParams.MAX_NEW_TOKENS: 800,
    GenParams.TEMPERATURE: 0.9,
    GenParams.TOP_P: 0.95,
    GenParams.REPETITION_PENALTY: 1.2
},
                credentials=credentials,
                project_id=project_id
            )

            response = model.generate(prompt=full_prompt)
            story = response['results'][0]['generated_text']

        except Exception as e:
            story = f"❌ Error generating story: {e}"

    return render_template("index.html", story=story)

if __name__ == "__main__":
    app.run(debug=True)
