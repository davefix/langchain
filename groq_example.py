"""
Simple Groq API example - uses GROQ_API_KEY from environment
"""

from groq import Groq

# Client reads GROQ_API_KEY automatically from environment
client = Groq()


def simple_chat(prompt: str) -> str:
    """Send a simple chat message and get response"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.7,
    )
    return response.choices[0].message.content


def streaming_chat(prompt: str):
    """Stream the response token by token"""
    stream = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        max_tokens=1024,
        temperature=0.7,
    )

    print("Response: ", end="", flush=True)
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()  # New line at the end


def multi_turn_conversation():
    """Example of a multi-turn conversation"""
    messages = [
        {"role": "user", "content": "What are the three laws of robotics?"},
    ]

    # First response
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile", messages=messages, max_tokens=1024
    )

    assistant_response = response.choices[0].message.content
    print(f"Assistant: {assistant_response}\n")

    # Add assistant response to conversation
    messages.append({"role": "assistant", "content": assistant_response})

    # Follow-up question
    messages.append({"role": "user", "content": "Who created these laws?"})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile", messages=messages, max_tokens=1024
    )

    print(f"Assistant: {response.choices[0].message.content}")


if __name__ == "__main__":
    print("=== Simple Chat Example ===")
    result = simple_chat("Explain quantum computing in one paragraph")
    print(f"Response: {result}\n")

    print("\n=== Streaming Example ===")
    streaming_chat("Write a haiku about coding")

    print("\n=== Multi-Turn Conversation ===")
    multi_turn_conversation()
