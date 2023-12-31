import openai

from .chat_interface import ChatInterface


class OpenAIChatBot(ChatInterface):

    def __init__(self):
        self.api_key = None
        self.model = None
        self.temperature = None
        self.messages = None
        self.system_message = None

    def init(self, config):
        self.api_key = config.get('api_key', '')
        self.model = config.get('model', 'gpt-3.5-turbo-16k')
        self.temperature = config.get('temperature', 1)
        self.system_message = config.get('system_message', self.system_message)
        self.messages = [{"role": "system", "content": self.system_message}]

    def chat(self, user_message, once=True):
        if once:
            self.messages = [{"role": "system", "content": self.system_message}]
            self.messages.append({"role": "user", "content": user_message})
        else:
            self.messages.append({"role": "user", "content": user_message})

        openai.api_key = self.api_key
        completion = openai.ChatCompletion.create(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature
        )

        return completion.choices[0].message.content


"""
def num_tokens_from_messages(messages, model="gpt-3.5-turbo"):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        num_tokens = 0
        for message in messages:
            num_tokens += 4
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += -1
        num_tokens += 2
        return num_tokens
    else:
        raise NotImplementedError(fnum_tokens_from_messages() is not presently implemented for model {model}.)
"""
