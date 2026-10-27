def get_user_query(prompt):
    # simplest way:
    # prompt ends with " ASSISTANT:" or something.
    # The last user message is before that.
    user_part = prompt.split(" ASSISTANT:")[-2] if " ASSISTANT:" in prompt else prompt
    user_part = user_part.split("USER:")[-1] if "USER:" in user_part else user_part
    return user_part.strip()
print(get_user_query("USER: Where are you? ASSISTANT:"))
