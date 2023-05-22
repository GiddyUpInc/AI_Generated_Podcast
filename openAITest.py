import os
import openai

def main():
    ## INSERT API KEY BELOW
    ## openai.api_key = .....
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="find 10 korean techno tracks",
        temperature=0.6,
    )
    print(response)
    

if __name__ == "__main__":
    main()