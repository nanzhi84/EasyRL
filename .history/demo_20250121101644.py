# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI

client = OpenAI(api_key="sk-5d532367be024f05818a953372c34bf4", base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False
)

print(response.choices[0].message.content)

curl https://api.deepseek.com/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer sk-5d532367be024f05818a953372c34bf4" -d "{ \"messages\": [ { \"role\": \"system\", \"content\": \"You are a test assistant.\" }, { \"role\": \"user\", \"content\": \"Testing. Just say hi and nothing else.\" } ], \"model\": \"deepseek-reasoner\" }"