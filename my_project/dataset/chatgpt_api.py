import os
import openai

import os
os.environ["http_proxy"] = "http://127.0.0.1:27999"
os.environ["https_proxy"] = "http://127.0.0.1:27999"

from openai import OpenAI

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="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",
)
#
# chat_completion = client.chat.completions.create(
#     messages=[
#         {
#             "role": "user",
#             "content": "Say this is a test",
#         }
#     ],
#     model="gpt-3.5-turbo",
# )
#
# response = chat_completion.get('choices')[0].get('message').get('content')
# print("模型的回复:", response)

instruction: str = "instruction: ly stupid and possibly dangerous little children in Indiana's Central High School, will soon be played by the"
prefix = ("Help me tune my prompt (the instruction) to get a better response while maintaining the original meaning of the instruction and user intent.",
          )

# Non-streaming:
print("----- standard request -----")
completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "user",
            "content": f"{instruction + prefix[0]}",
        },
    ],
)
print(completion.choices[0].message.content)