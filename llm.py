from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq()


def talk(text):
    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {
                "role": "system",
                "content": "Only give me a score out of 500, return only NUMBER",
            },
            {"role": "user", "content": text},
        ],
        temperature=0,
        max_completion_tokens=1024,
        top_p=0,
        stream=True,
        stop=None,
    )

    res = ""
    for chunk in completion:
        res += chunk.choices[0].delta.content or ""

    return res
