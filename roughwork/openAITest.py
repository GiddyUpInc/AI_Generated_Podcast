import os
import json
import openai

def main():
    ## INSERT API KEY BELOW
    openai.api_key = "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"
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="""Write the transcript for a long form podcast between Joe Rogan and an baby preparing for a moonlanding, but Joe Rogan is very sceptical. Format it in this style:\n
        \n Host: \"\"
        \n Guest: \"\"
        \n Host: \"\" """,
        temperature=0.9,
        max_tokens=1000,
    )
    response = json.loads(str(response))
    print(response["choices"][0]["text"])
    

if __name__ == "__main__":
    main()