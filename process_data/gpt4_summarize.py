input_dir = "<path_to_your_metadata_json_files>"  # Replace with the path to your folder containing metadata (.json) files
base_url = "<your_base_url>"  # Replace with the base URL for the API
api_key = "<your_api_key>"  # Replace with your API key
model = "<your_model>"  # Replace with your model name

import os
import json
import random
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(base_url=base_url, api_key=api_key)

def log_error(file_path, error_message):
    """Logs error messages to a specified log file."""
    os.makedirs("logs", exist_ok=True)
    with open("logs/gpt4_summarize_error_log.txt", 'a', encoding='utf-8') as log_file:
        log_file.write(f"Error processing {file_path}: {error_message}\n")

def process_json(metadata, language):
    """
    Processes the given metadata of a music piece using GPT-4 API. 

    This function sends the metadata and target language to the GPT-4 model to generate 
    a structured summary. The summary is provided in both English and the specified 
    non-English language from the 'nen_language' field. 

    If the provided metadata lacks sufficient music-related details, the function returns `None`.

    Parameters:
    - metadata (dict): A dictionary containing the metadata of the music piece.
    - language (str): The target non-English language for the summary.

    Returns:
    - str: A JSON-formatted string containing the English and non-English summaries, 
      or `None` if there is insufficient information.
    """
    system = """Your task is to provide a concise, comprehensive, and coherent summary of the music piece using the provided metadata. Please write the summary in English first, and then write an equivalent summary in the specified non-English language from the "nen_language" field. Use this JSON format:
{
  "summary_en": "Your English summary here.",
  "summary_nen": {
    "language": "Specified non-English language.",
    "summary": "Your non-English summary here."
  }
If there is not enough music-related information, return `None` instead.
}
"""
    user1 = """{
    "title": "Brejeiro",
    "composer": "Ernesto Nazareth",
    "genres": ["Choro", "Classical", "Instrumental"],
    "description": "\"Brejeiro\" is in A major and 2/4 time. A joyful melody begins at bar six, and a lively tango rhythm starts at bar fourteen. It has a D.C. al Fine at bar fifty-three and ends on two quarter notes in bar thirty-seven. The piece, with its vibrant melodies and rhythms, reflects celebration and carefreeness, embodying the spirit of Brazilian music.",
    "tags": ["Brazilian", "Choro", "Piano"],
    "ensembles": ["Solo Piano", "Small Ensemble"],
    "instruments": ["Piano"],
    "nen_language": "Japanese"
}
"""
    assistant1 = """{
    "summary_en": "Brejeiro, composed by Ernesto Nazareth, is a lively choro piece in A major and 2/4 time. It features a joyful melody that begins at bar six and a vibrant tango rhythm introduced at bar fourteen. The piece includes a D.C. al Fine at bar fifty-three, concluding on two quarter notes in bar thirty-seven. With its themes of celebration and carefreeness, Brejeiro beautifully captures the essence of Brazilian music and is well-suited for solo piano and small ensembles.",
    "summary_nen": {
    "language": "Japanese",
    "summary": "「ブレジェイロ」は、エルネスト・ナザレが作曲した活気あふれるショーロの作品で、イ長調の2/4拍子で書かれています。第6小節から始まる喜びに満ちたメロディーと、第14小節で導入される活気あるタンゴのリズムが特徴です。この曲には、第53小節でのD.C. al Fineが含まれ、また第37小節で二つの四分音符で締めくくられています。「ブレジェイロ」は、お祝いと無邪気さのテーマを持ち、ブラジル音楽の本質を美しく捉えており、ソロピアノや小編成のアンサンブルにぴったりの作品です。"
  }
}
"""
    user2 = """{
    "title": "Untitled",
    "composer": "Unknown",
    "description": "This is a good song.",
    "nen_language": "Russian"
}
"""
    assistant2 = "None"
    filepaths = metadata.pop('filepaths')
    metadata = {k: v for k, v in metadata.items() if v is not None}

    metadata["nen_language"] = language
    metadata = json.dumps(metadata, ensure_ascii=False, indent=4)
    summaries = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user1},
            {"role": "assistant", "content": assistant1},
            {"role": "user", "content": user2},
            {"role": "assistant", "content": assistant2},
            {"role": "user", "content": metadata},
        ]
    ).choices[0].message.content

    if summaries == "None":
        raise ValueError("Received 'None' as summaries response")
    
    metadata = json.loads(metadata)
    summaries = json.loads(summaries)

    if metadata["nen_language"] == summaries["summary_nen"]["language"]:
        metadata.pop("nen_language")
        metadata["summary_en"] = summaries["summary_en"]
        metadata["summary_nen"] = summaries["summary_nen"]
        metadata["filepaths"] = filepaths
        return metadata
    else:
        raise ValueError("Language mismatch: nen_language does not match summary_nen language")

def process_files(input_dir):
    # Create output directory with _summarized suffix
    output_dir = input_dir + "_summarized"
    
    # Define available languages
    languages = """Afrikaans
                    Amharic
                    Arabic
                    Assamese
                    Azerbaijani
                    Belarusian
                    Bulgarian
                    Bengali
                    Bengali (Romanized)
                    Breton
                    Bosnian
                    Catalan
                    Czech
                    Welsh
                    Danish
                    German
                    Greek
                    Esperanto
                    Spanish
                    Estonian
                    Basque
                    Persian
                    Finnish
                    French
                    Western Frisian
                    Irish
                    Scottish Gaelic
                    Galician
                    Gujarati
                    Hausa
                    Hebrew
                    Hindi
                    Hindi (Romanized)
                    Croatian
                    Hungarian
                    Armenian
                    Indonesian
                    Icelandic
                    Italian
                    Japanese
                    Javanese
                    Georgian
                    Kazakh
                    Khmer
                    Kannada
                    Korean
                    Kurdish (Kurmanji)
                    Kyrgyz
                    Latin
                    Lao
                    Lithuanian
                    Latvian
                    Malagasy
                    Macedonian
                    Malayalam
                    Mongolian
                    Marathi
                    Malay
                    Burmese
                    Burmese (Romanized)
                    Nepali
                    Dutch
                    Norwegian
                    Oromo
                    Oriya
                    Punjabi
                    Polish
                    Pashto
                    Portuguese
                    Romanian
                    Russian
                    Sanskrit
                    Sindhi
                    Sinhala
                    Slovak
                    Slovenian
                    Somali
                    Albanian
                    Serbian
                    Sundanese
                    Swedish
                    Swahili
                    Tamil
                    Tamil (Romanized)
                    Telugu
                    Telugu (Romanized)
                    Thai
                    Filipino
                    Turkish
                    Uyghur
                    Ukrainian
                    Urdu
                    Urdu (Romanized)
                    Uzbek
                    Vietnamese
                    Xhosa
                    Yiddish
                    Chinese (Simplified)
                    Chinese (Traditional)
                    Cantonese"""
    languages = [language.strip() for language in languages.split("\n")]

    # Walk through the input directory
    for root, _, files in os.walk(input_dir):
        # Construct the corresponding path in the output folder
        relative_path = os.path.relpath(root, input_dir)
        output_path = os.path.join(output_dir, relative_path)

        # Create the output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)

        for file in files:
            if file.endswith('.json'):
                input_file = os.path.join(root, file)
                output_file = os.path.join(output_path, file)

                try:
                    # Read the JSON file
                    with open(input_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)

                    # Randomly select a language from the list of languages
                    language = random.choice(languages)

                    # Process the JSON data
                    processed_metadata = process_json(metadata, language)

                    # Write the processed JSON to the output file
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(processed_metadata, f, indent=4, ensure_ascii=False)

                    print(f"Processed: {input_file} -> {output_file}")

                except Exception as e:
                    print(f"Failed to process {input_file}: {e}")
                    log_error(input_file, str(e))

if __name__ == "__main__":
    process_files(input_dir)
