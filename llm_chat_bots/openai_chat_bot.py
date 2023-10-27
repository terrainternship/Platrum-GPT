import openai
import tiktoken
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)

from .chat_interface import ChatInterface


class OpenAIChatBot(ChatInterface):

    def __init__(self):
        self.prompt = None
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
        self.prompt = config.get('prompt', '')

    @retry(
        retry=retry_if_exception_type((
                openai.error.APIError,
                openai.error.APIConnectionError,
                openai.error.RateLimitError,
                openai.error.ServiceUnavailableError,
                openai.error.Timeout
        )),
        wait=wait_random_exponential(multiplier=1, max=60),
        stop=stop_after_attempt(10)
    )
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

    @retry(
        retry=retry_if_exception_type((
                openai.error.APIError,
                openai.error.APIConnectionError,
                openai.error.RateLimitError,
                openai.error.ServiceUnavailableError,
                openai.error.Timeout
        )),
        wait=wait_random_exponential(multiplier=1, max=60),
        stop=stop_after_attempt(10)
    )
    def completion(self, user_message):
        if user_message:
            self.prompt += user_message

        messages = [{"text": self.prompt}]
        calc_tokens = num_tokens_from_messages(messages)
        if calc_tokens >= 4097:
            return ""

        openai.api_key = self.api_key
        completion = openai.Completion.create(
                model=self.model,
                prompt=self.prompt,
                max_tokens=(4097 - calc_tokens),
                temperature=self.temperature
        )

        return completion.choices[0].text


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
