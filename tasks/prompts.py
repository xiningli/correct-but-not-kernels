"""K1–K4: fixed single-turn prompt with the pinned reference; no held-out tuning."""
from pathlib import Path
from env.tasks import ROOT

def messages(task):
    contract = (ROOT/'tasks/prompt.md').read_text()
    return [{'role':'user', 'content':contract + '\n\nReference:\n```python\n' + task.source() + '\n```'}]

def prompt_token_ids(tokenizer, conversation):
    """K1/K4: pin the token-list API; Transformers 5.17 defaults to BatchEncoding."""
    return tokenizer.apply_chat_template(conversation, tokenize=True,
        add_generation_prompt=True, enable_thinking=False, return_dict=False)
