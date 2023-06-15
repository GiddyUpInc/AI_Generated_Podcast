import os
import json
import torch
import openai
from textToSpeech import TTS

## INSERT API KEY BELOW
openai.api_key = "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"

def generate_script() -> list:
    # Make api call
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="""Write the transcript for a long form podcast between Joe Rogan and an baby preparing for a moonlanding, but Joe Rogan is very sceptical. Format it in this style:\n
        \n Host: \"\"
        \n Guest: \"\"
        \n Host: \"\" """,
        temperature=0.9,
        max_tokens=1000,
    )
    
    # Parse Response
    response = json.loads(str(response))
    script_string = response["choices"][0]["text"]
    print(script_string)

    script_list = script_string.split("\n\n")
    for i, line in enumerate(script_list):
        if ":" in line:
            script_list[i] = line.split(":")[1]
    return script_list

def main():
    tts = TTS()
    voice1,voice2 = tts.get_embeddings()
    script = generate_script()
    print(script)
    speech = torch.empty(1, dtype=torch.float32)
    for i, line in enumerate(script):
        if i % 2:
            voice = voice1
        else:
            voice = voice2

        new_speech = tts.generate_speech(line,voice)
        speech = torch.cat((speech, new_speech))
    tts.save_audio(speech)

if __name__ == "__main__":
    main()